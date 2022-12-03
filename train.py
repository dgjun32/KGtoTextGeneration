import os
import torch
import time
import numpy as np
import pickle
import torch.nn as nn
from logging import getLogger
from dataset import Vocab, NLP, S2SDataset
from utils import build_optimizer, init_seed, init_logger, init_device, read_configuration, collate_fn_graph_text, \
    format_time
from model import GraphEncoder, GraphPointer, GraphReconstructor, compute_ce_loss, compute_alignment_loss
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import Dataset, DataLoader


# compute training loss for batch
def train_batch_loss(bart, graph_enc, copyer, batch, device):
    batch_size=20

    # train mode
    graph_enc.train()
    bart.train()

    # compute alignment loss (agg_emb is derived from teacher emb)
    align_loss, agg_emb, teacher_emb = compute_alignment_loss(batch, bart, graph_enc, device, train=True)
    
    # compute generation loss
    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
        recon_relations, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks = batch

    decoder_input_ids = gen_outputs[:,:-1].to(device)
    
    ## create decoder labels
    labels = gen_outputs[:,1:].contiguous().to(device)
    
    ## create decoder attention mask
    decoder_mask = gen_masks[:,:-1].to(device)
    cross_mask = torch.ones(batch_size, 10, dtype=torch.long).to(device)
    
    output = bart(input_ids=None,
                  #inputs_embeds=agg_emb,
                  encoder_outputs=[agg_emb],
                  decoder_input_ids=decoder_input_ids,
                  labels = labels,
                  attention_mask = cross_mask,
                  decoder_attention_mask=decoder_mask,
                  output_hidden_states=True,
                  return_dict = True)
    gen_loss = output[0]
    
    # compute copy loss
    with torch.no_grad():
        decoder_input_embeddings = bart.get_input_embeddings()(gen_outputs[:, :-1].to(device))
    decoder_output_hiddens = output["decoder_hidden_states"][-1]
    pointer = pointer.to(device)
    pointer_masks = pointer_masks.to(device)
    copy_prob = copyer(decoder_input_embeddings, decoder_output_hiddens, pointer[:, 1:])
    copy_loss = copy_prob.masked_select(pointer_masks[:, 1:]).mean()
    
    # compute reconstruction loss
    #recon_positions = recon_positions.to(device)
    #recon_relations = recon_relations.to(device)
    #recon_masks = recon_masks.to(device)
    #rec_logits = reconstructor(recon_positions, output['encoder_last_hidden_state'])
    #rec_loss = compute_ce_loss(rec_logits, recon_relations, recon_masks)


    return align_loss, gen_loss, copy_loss


# compute validation loss for batch
def val_batch_loss(bart, graph_enc, batch, device):
    batch_size = 4

    #eval mode
    graph_enc.eval()
    bart.eval()

    # compute alignment loss

    with torch.no_grad():
        align_loss, agg_emb, _ = compute_alignment_loss(batch, bart, graph_enc, device, train=False)
    
    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
        recon_relations, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks = batch
    
    # create decoder input embeds (concatenated with gprompt)
    #with torch.no_grad():
    #    decoder_inputs_embeds = bart.get_input_embeddings()(gen_outputs[:,:-1].to(device))
    #gprompt = graph_enc.gprompt.repeat(decoder_inputs_embeds.shape[0], 1, 1).to(device)
    #decoder_inputs_embeds = torch.cat([gprompt, decoder_inputs_embeds], dim=1) # (batch_size, num_prompts+seq_len, 768)
    decoder_input_ids = gen_outputs[:,:-1].to(device)

    labels = gen_outputs[:,1:].contiguous().to(device)
    #labels = torch.cat([torch.zeros(batch_size, 10, dtype=torch.long).to(device), labels], dim=1)
    
    ## create decoder attention mask
    #bos = torch.ones(decoder_inputs_embeds.shape[0], 10, dtype=torch.long).to(device)
    #mask = torch.cat([bos, gen_masks[:,:-1].to(device)], dim=1)
    decoder_mask = gen_masks[:,:-1].to(device)
    cross_mask = torch.ones(batch_size, 10, dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = bart(input_ids = None,
                    encoder_outputs = [agg_emb],
                    #inputs_embeds = agg_emb,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels,
                    attention_mask = cross_mask,  # cross attn mask
                    decoder_attention_mask=decoder_mask) # decoder attn mask
    
    gen_loss = output[0]
    #logits = output.logits
    #real_logits = logits[:,10:,:]
    #gen_loss = nn.CrossEntropyLoss()(real_logits.reshape(-1, 50265), labels.reshape(-1))
    
    return align_loss, gen_loss


def generate(bart, graph_enc, batch, tokenizer, device):

    bart.eval()
    graph_enc.eval()

    generated_text, reference_text = [], []
    batch_size = 4

    nodes, edges, types, node_masks, kd_description, kd_description_masks, kd_positions, \
        recon_relations, recon_positions, recon_masks, gen_outputs, gen_masks, pointer, pointer_masks = batch

    # compute student embedding
    with torch.no_grad():
        _, agg_emb, _ = compute_alignment_loss(batch, bart, graph_enc, device, train=False)

    # extract learned graph prompt
    #gprompt = graph_enc.gprompt.repeat(batch_size, 1, 1).to(device) # (batch_size x num_prompts x 768)
    #bos_prompt = bart.get_input_embeddings()(torch.zeros(batch_size,1, dtype=torch.long).to(device)) # (batch_size x 1 x 768)
    #prompt = torch.cat([gprompt, bos_prompt], dim=1) # (batch_size x num_prompts+1 x 768)

    # create encoder outputs
    encoder_outputs = BaseModelOutput()
    encoder_outputs.last_hidden_state = agg_emb # (batch_size, num_entities, 768)

    generated_ids = bart.generate(
                                    #decoder_input_ids=torch.zeros(batch_size, 1, dtype=torch.long).to(device),
                                    encoder_outputs=encoder_outputs,
                                    #attention_mask=node_masks.to(device),
                                    num_beams=8,
                                    max_length=128,
                                    early_stopping=True)

    generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    reference = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
    generated_text.extend(generated)
    reference_text.extend(reference)
    
    return generated_text, reference_text


def main():
    # set seed
    torch.manual_seed(2022)
    np.random.seed(2022)
    torch.cuda.manual_seed(2022)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:4')
    print('device : {}'.format(device))
    # build vocab & dataloader
    vocabs = dict()
    vocabs["node"] = Vocab("data/webnlg-few/node.pkl")
    vocabs["relation"] = Vocab("data/webnlg-few/relation.pkl")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    train_dataset = S2SDataset(data_dir='data/',
                     dataset='webnlg-few',
                     tokenizer=tokenizer,
                     node_vocab=vocabs['node'],
                     relation_vocab=vocabs['relation'],
                     num_samples=500,
                     usage='train')
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=20,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=True,
                                            collate_fn=collate_fn_graph_text)

    val_dataset = S2SDataset(data_dir='data/',
                     dataset='webnlg-few',
                     tokenizer=tokenizer,
                     node_vocab=vocabs['node'],
                     relation_vocab=vocabs['relation'],
                     num_samples=100,
                     usage='valid')
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=4,
                                            shuffle=False,
                                            drop_last=False,
                                            pin_memory=True,
                                            collate_fn=collate_fn_graph_text)

    # build bart model
    bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn", forced_bos_token_id=0)
    bart = bart.to(device)

    # freeze parameter of bart encoder model

    #for param in bart.parameters():
    #    param.requires_grad = False
    
    for n, param in bart.named_parameters():
        if 'decoder' in n:
            param.requires_grad = False
        #if 'embed' in n:
        #    param.requires_grad = False
        #if 'shared' in n:
        #    param.requires_grad = False
        
        
    
    # build graph encoder
    graph_enc = GraphEncoder(num_nodes = vocabs['node'].size(), # 존재하는 node의 총 개수
                         num_relations = vocabs['relation'].size(), # 존재하는 relation의 총 개수 
                         gnn_layers = 2,
                         embedding_size = 1024,
                         initilized_embedding='data/webnlg-few/node_embeddings.npy',
                         device=device,
                         dropout_ratio=0.3)
    graph_enc = graph_enc.to(device)

    # build copyer
    copyer = GraphPointer(1024, 1024)
    copyer.to(device)

    # build reconstructor
    #reconstructor = GraphReconstructor(vocabs["relation"].size(), 1024)
    #reconstructor.to(device)
    
    # optimizer
    ext_parameters = []
    for p in graph_enc.parameters():
        if p.requires_grad:
            ext_parameters.append(p)
    for p in copyer.parameters():
        if p.requires_grad:
            ext_parameters.append(p)
    #for p in reconstructor.parameters():
    #    if p.requires_grad:
    #        ext_parameters.append(p)
    
    #lm_parameters = []
    #for p in bart.parameters():
    #    if p.requires_grad:
    #        lm_parameters.append(p)
    
    ext_optimizer = torch.optim.AdamW(ext_parameters, lr=0.0001)
    #lm_optimizer = torch.optim.AdamW(lm_parameters, lr=0.000001)

    trainables = []
    for n, param in bart.named_parameters():
        if param.requires_grad:
            trainables.append(n)
    for n, param in graph_enc.named_parameters():
        if param.requires_grad:
            trainables.append(n)
    for n, param in copyer.named_parameters():
        if param.requires_grad:
            trainables.append(n)
    #for n, param in graph_enc.named_parameters():
    #    if param.requires_grad:
    #        trainables.append(n)
    
    print('Trainable Parameters : {}'.format(trainables))

    print("Start Training")
    for epoch in range(200):
        print('-'*20 + 'Epoch {}'.format(epoch+1) + '-'*20)
        batch_size=20
        epoch_loss = 0
        epoch_align_loss = 0
        epoch_gen_loss = 0
        epoch_copy_loss = 0
        #epoch_rec_loss = 0
        val_loss = 0
        val_align_loss=0
        val_gen_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            align_loss, gen_loss, copy_loss = train_batch_loss(bart, graph_enc, copyer, batch, device)
            loss = 0.7*align_loss + gen_loss + 0.5*copy_loss

            # backward
            ext_optimizer.zero_grad()
            #lm_optimizer.zero_grad()
            loss.backward()
            ext_optimizer.step()
            #lm_optimizer.step()

            # verbosity
            epoch_loss += loss.item()
            epoch_align_loss += align_loss.item()
            epoch_gen_loss += gen_loss.item()
            #epoch_rec_loss += rec_loss.item()
            epoch_copy_loss += copy_loss.item()

            cur_loss = epoch_loss / (step+1) / batch_size
            cur_align_loss = epoch_align_loss / (step+1) / batch_size
            cur_gen_loss = epoch_gen_loss / (step+1) / batch_size
            #cur_rec_loss = epoch_rec_loss / (step+1) / batch_size
            cur_copy_loss = epoch_copy_loss / (step+1) / batch_size

            if (step+1)%1 == 0:
                print('{} / {} | total loss : {} | gen loss : {} | align_loss : {} | copy loss : {}'.format(
                    step+1, len(train_dataloader), cur_loss, cur_gen_loss, cur_align_loss, cur_copy_loss))
        
        print('-'*50)
        # compute val loss at epoch end
        for val_idx, batch in enumerate(val_dataloader):
            
            # compute validation loss
            a_loss, g_loss = val_batch_loss(bart, graph_enc, batch, device)
            val_align_loss += a_loss.item()
            val_gen_loss += g_loss.item()
            val_loss += a_loss.item() + g_loss.item()
            
            # generated sample sequence
            if val_idx == 0:
                gen_text, ref_text = generate(bart, graph_enc, batch, tokenizer, device)
        
        val_loss = val_loss / batch_size / len(val_dataloader)
        val_align_loss = val_align_loss / batch_size / len(val_dataloader)
        val_gen_loss = val_gen_loss / batch_size / len(val_dataloader)

        # verbose validation loss
        print('Validation Epoch {} | total loss : {} | gen loss : {} | align loss : {}'.format(
            epoch+1, val_loss, val_gen_loss, val_align_loss
        ))
        # verbose sample sequence
        print('Gen : {} \n Ref : {} \n'.format(gen_text[2], ref_text[2]))


        # save checkpoint 
        if (epoch+1) % 5 == 0:
            torch.save(graph_enc.state_dict, 'ckpt/epoch_{}.pt'.format(epoch+1))

            
if __name__ == '__main__':
    main()