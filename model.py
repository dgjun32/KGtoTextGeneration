

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from transformers import RobertaTokenizer, RobertaForMaskedLM
import math
import numpy as np
import torch.backends.cudnn as cudnn
import transformers
from transformers import BartForConditionalGeneration


class ListModule(nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Model initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)


class GraphEncoder(nn.Module):
    def __init__(self, num_nodes, num_relations, gnn_layers, embedding_size, initilized_embedding, device, dropout_ratio=0.3):
        super(GraphEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.gnn_layers = gnn_layers
        self.embedding_size = embedding_size
        self.dropout_ratio = dropout_ratio

        self.node_embedding = nn.Embedding(num_nodes, embedding_size)
        self.node_embedding.from_pretrained(torch.from_numpy(np.load(initilized_embedding)), freeze=False)
        self.device = device
        self.dropout = nn.Dropout(dropout_ratio)

        self.gnn = []
        for layer in range(gnn_layers):
            self.gnn.append(RGCNConv(embedding_size, embedding_size, num_relations))  # if rgcn is too slow, you can use gcn
        self.gnn = ListModule(*self.gnn)
        # learnable prompt for decoder cross attention (graph prompt)
        bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0)
        with torch.no_grad():
            #gprompt = bart.get_input_embeddings()(torch.zeros(10, dtype=torch.long))
            gprompt = torch.randn(10, 1024)*0.02
        self.gprompt = nn.Parameter(gprompt, requires_grad = True) # prompts concatenated to 
        self.resampler = nn.MultiheadAttention(1024, 8, batch_first=True)

    def forward(self, nodes, edges, types, teacher_emb = None):
        """
        :param nodes: tensor, shape [batch_size, num_nodes]
        :param edges: List(List(edge_idx)) : list length of batch size
        :param types: List(type_idx)
        """
        batch_size = nodes.size(0)
        device = nodes.device

        # (batch_size, num_nodes, output_size)
        node_embeddings = []
        for bid in range(batch_size):
            embed = self.node_embedding(nodes[bid, :]).to(device)
            edge_index = torch.as_tensor(edges[bid], dtype=torch.long, device=device)
            edge_type = torch.as_tensor(types[bid], dtype=torch.long, device=device)
            for lidx, rgcn in enumerate(self.gnn):
                if lidx == len(self.gnn) - 1:
                    embed = rgcn(embed, edge_index=edge_index, edge_type=edge_type)
                else:
                    embed = self.dropout(F.relu(rgcn(embed, edge_index=edge_index, edge_type=edge_type)))
            node_embeddings.append(embed)
        node_embeddings = torch.stack(node_embeddings, 0)  # [batch_size, num_node, embedding_size]
        gprompt = self.gprompt.repeat(batch_size,1,1)
        #prompt = torch.cat([node_embeddings, self.gprompt.repeat(batch_size, 1, 1)], dim=1) # [batch_size, num_node+K, emb_dim]
        #mask = torch.ones(prompt.shape[1], prompt.shape[1])
        #mask[-10:, :] = 0

        if teacher_emb is None: # inference
            aggregated = self.resampler(gprompt, node_embeddings, node_embeddings)[0] #attn_mask = mask.to(self.device))[0]
            agg_embeddings = aggregated # (batch_size, K, emb_dim)
            
        else: # training
            aggregated = self.resampler(gprompt, teacher_emb, teacher_emb)[0] #attn_mask = mask.to(self.device))[0]
            agg_embeddings = aggregated

        return node_embeddings, agg_embeddings

class GraphReconstructor(nn.Module):
    def __init__(self, num_relations, hidden_size):
        super(GraphReconstructor, self).__init__()
        self.num_relations = num_relations
        self.hidden_size = hidden_size

        self.proj_linear = nn.Linear(3 * hidden_size, num_relations)

    def forward(self, pairs, hidden_states):
        """
        :param pairs: tensor [batch_size, num_pairs, 2, 2]
        :param hidden_states: tensor [batch_size, seq_len, hidden_size]
        """
        batch_size, num_pairs = pairs.size(0), pairs.size(1)
        hidden_size = hidden_states.size(-1)

        head, tail = torch.chunk(pairs, chunks=2, dim=2)

        h_start, h_end = torch.chunk(head, chunks=2, dim=3)
        t_start, t_end = torch.chunk(tail, chunks=2, dim=3)

        hs_expand = h_start.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        hs_embed = torch.gather(hidden_states, dim=1, index=hs_expand)

        he_expand = h_end.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        he_embed = torch.gather(hidden_states, dim=1, index=he_expand)

        head_embed = (hs_embed + he_embed) / 2.0

        ts_expand = t_start.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        ts_embed = torch.gather(hidden_states, dim=1, index=ts_expand)

        te_expand = t_end.contiguous().view(batch_size, num_pairs).unsqueeze(-1).expand(-1, -1, hidden_size)
        te_embed = torch.gather(hidden_states, dim=1, index=te_expand)

        tail_embed = (ts_embed + te_embed) / 2.0

        logits = self.proj_linear(torch.cat([head_embed, tail_embed, head_embed * tail_embed], dim=-1))

        return logits


class GraphPointer(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(GraphPointer, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.proj_linear = nn.Linear(embedding_size + hidden_size, 1)

    def forward(self, embeddings, hidden_states, pointer):
        """
        :param embeddings: tensor [batch_size, seq_len, embedding_size]
        :param hidden_states: tensor [batch_size, seq_len, hidden_size]
        :param pointer: tensor [batch_size, seq_len]
        """
        copy_prob = torch.sigmoid(self.proj_linear(torch.cat([embeddings, hidden_states], dim=-1))).squeeze(-1)
        copy_prob = torch.where(pointer.bool(), 1 - copy_prob, copy_prob)

        return copy_prob

    
    
def compute_kd_loss(node_embeddings, desc_embeddings, node_masks, kd_masks):
    assert node_embeddings.size() == desc_embeddings.size()
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(node_embeddings, desc_embeddings)
    loss = loss.mean(dim=-1)
    masks = node_masks * kd_masks
    loss = loss.masked_select(masks).mean()
    return loss

def compute_ce_loss(logits, labels, masks):
    ce_loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss = loss.reshape_as(labels)
    loss = loss.masked_select(masks).mean()
    return loss

def compute_alignment_loss(batch, bart, graph_enc, device, train=True):
    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
        recon_relations, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks = batch
    with torch.no_grad():
        output_dict = bart(input_ids=kd_description.to(device),
                          attention_mask=kd_description_masks.to(device),
                          output_hidden_states=True,
                          return_dict=True)
    positions = kd_positions.unsqueeze(-1).expand(-1, -1, output_dict["encoder_last_hidden_state"].size(-1)).to(device)
    teacher_embeddings = torch.gather(output_dict["encoder_last_hidden_state"], dim=1, index=positions).to(device)
    teacher_embeddings = teacher_embeddings.detach()
    if train:
        # agg_embeddings is made up of teacher embeddings
        node_embeddings, agg_embeddings = graph_enc(nodes.to(device), edges, types, teacher_embeddings)
    else:
        node_embeddings, agg_embeddings = graph_enc(nodes.to(device), edges, types)
    node_masks = node_masks.to(device)
    kd_masks = torch.ne(kd_positions, 0).to(device)
    kd_loss = compute_kd_loss(node_embeddings, teacher_embeddings, node_masks, kd_masks)
    return kd_loss, agg_embeddings, teacher_embeddings

