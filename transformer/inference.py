import torch
import torch.nn as nn
import numpy as np
from model import Transformer
import data_utils

import os

from config import MyConfig

def process(sentence):
    tokens = sentence.split(' ') + ['<eos>']
    enc_tokens = tokens + ['<pad>'] * (max_len - len(tokens))            
    dec_tokens = [start_tokens] + ['<pad>'] * (max_len - 1)
    return enc_tokens, dec_tokens


if __name__ == '__main__':
    config = MyConfig('config/config.ini')
    
    train_data_path = config.train_data_path
    model_entire_path = config.model_entire_path
    model_trace_path = config.model_trace_path
    
    batch_size = config.batch_size
    max_len = config.max_len
    epochs = config.epochs
    PAD_VALUE = config.pad_value
    device = config.device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_hiddens = config.num_hiddens
    query_size = key_size = value_size = num_hiddens
    ffn_num_hiddens = config.ffn_num_hiddens
    num_heads = config.num_heads
    encoder_layers = config.encoder_layers
    decoder_layers = config.decoder_layers
    dropout = config.dropout
    use_bias = config.use_bias
    
    
    corpus = data_utils.Corpus(train_data_path)
    train_iter, source_vocab, target_vocab = corpus.get_dataset(batch_size=batch_size, num_steps=max_len, num_examples=None, shuffle=False)
    source_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)
    
    start_tokens = '<bos>'
    end_tokens = '<eos>'
    
    # inference 1
    print(f'loading {model_entire_path}')
    model_entire = torch.load(model_entire_path)
    model_entire = model_entire.to(device)
    model_entire.eval()

    # inference 2
    print(f'loading {model_trace_path}')
    model_trace = torch.jit.load(model_trace_path)
    model_trace = model_trace.to(device)
    model_trace.eval()
    
    sentence = ''
    while True:
        sentence = input('input:')
        # sentence = 'i am busy .'
        if sentence == 'q':
            break
        
        # inference 1
        enc_tokens, dec_tokens = process(sentence)
        idx = 0
        next_tokens = start_tokens
        pred_tokens = []

        enc_inputs = torch.tensor(source_vocab[enc_tokens], dtype=torch.long, device=device).reshape((1, -1))
        enc_self_attn_mask = data_utils.get_attn_pad_mask(enc_inputs, enc_inputs, PAD_VALUE).to(device)    
        while next_tokens != end_tokens:
            dec_tokens[idx] = next_tokens
            dec_inputs = torch.tensor([target_vocab[dec_tokens]], dtype=torch.long, device=device).reshape((1, -1))
            dec_self_attn_mask = data_utils.get_attn_pad_mask(dec_inputs, dec_inputs, PAD_VALUE).to(device)
            dec_self_subsequent_attn_mask = data_utils.get_attn_subsequent_mask(dec_inputs)
            dec_self_attn_mask = dec_self_attn_mask + dec_self_subsequent_attn_mask
            dec_self_attn_mask = dec_self_attn_mask.gt(0).to(device)

            dec_enc_attn_mask = data_utils.get_attn_pad_mask(dec_inputs, enc_inputs, PAD_VALUE).to(device)
            
            dec_outputs = model_entire(enc_inputs, dec_inputs, enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask)
            dec_logit = dec_outputs.squeeze(0)
            dec_result = dec_logit.argmax(dim=-1)
            next_idx = dec_result[idx]
            next_tokens = target_vocab.to_token(next_idx)
            pred_tokens.append(next_tokens)
            idx += 1
            if idx >= max_len:
                break
        print(dec_logit)
        print('input:', ' '.join(enc_tokens))  
        print('pred:', ' '.join(pred_tokens))


        # inference 2
        enc_tokens, dec_tokens = process(sentence)
        next_tokens = start_tokens
        pred_tokens = []
        idx = 0
        
        enc_inputs = torch.tensor(source_vocab[enc_tokens], dtype=torch.long, device=device).reshape((1, -1))
        enc_self_attn_mask = data_utils.get_attn_pad_mask(enc_inputs, enc_inputs, PAD_VALUE).to(device)    
        while next_tokens != end_tokens:
            dec_tokens[idx] = next_tokens
            dec_inputs = torch.tensor([target_vocab[dec_tokens]], dtype=torch.long, device=device).reshape((1, -1))
            dec_self_attn_mask = data_utils.get_attn_pad_mask(dec_inputs, dec_inputs, 1).to(device)
            dec_self_subsequent_attn_mask = data_utils.get_attn_subsequent_mask(dec_inputs)
            dec_self_attn_mask = dec_self_attn_mask + dec_self_subsequent_attn_mask
            dec_self_attn_mask = dec_self_attn_mask.gt(0).to(device)

            dec_enc_attn_mask = data_utils.get_attn_pad_mask(dec_inputs, enc_inputs, 1).to(device)
            
            dec_outputs = model_trace(enc_inputs, dec_inputs, enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask)

            dec_logit = dec_outputs.squeeze(0)
            dec_result = dec_logit.argmax(dim=-1)
            next_idx = dec_result[idx]
            next_tokens = target_vocab.to_token(next_idx)
            pred_tokens.append(next_tokens)
            idx += 1
            if idx >= max_len:
                break
        print(dec_logit)
        print('input:', ' '.join(enc_tokens))  
        print('pred:', ' '.join(pred_tokens))



    ''' test '''
    # # inference 1
    # model_entire = torch.load('saved_models/temp.pt')
    # # model_entire = torch.load(model_entire_path)
    # model_entire = model_entire.to(device)
    # model_entire.eval()
    
    # model_trace = torch.jit.load('saved_models/temp_trace.pt')
    # # model_trace = torch.jit.load(model_trace_path)
    # model_trace = model_trace.to(device)
    # model_trace.eval()

    # sentence = ''
    # while True:
    #     sentence = input('input:')
    #     if sentence == 'q':
    #         break
    #     # sentence = 'i see .'
    #     enc_tokens, dec_tokens = process(sentence)
    #     next_tokens = start_tokens
    #     pred_tokens = []
    #     idx = 0

    #     enc_inputs = torch.tensor(source_vocab[enc_tokens], dtype=torch.long, device=device).reshape((1, -1))
    #     while next_tokens != end_tokens:
    #         dec_tokens[idx] = next_tokens
    #         dec_inputs = torch.tensor([target_vocab[dec_tokens]], dtype=torch.long, device=device).reshape((1, -1))
            
    #         dec_outputs = model_entire(enc_inputs, dec_inputs)
    #         dec_logit = dec_outputs.squeeze(0)
    #         dec_result = dec_logit.argmax(dim=-1)
    #         next_idx = dec_result[idx]
    #         next_tokens = target_vocab.to_token(next_idx)
    #         pred_tokens.append(next_tokens)
    #         idx += 1
    #         if idx >= max_len:
    #             break
    #         # print(dec_outputs.shape)
    #         # print(dec_logit.shape)
    #         # print(dec_result.shape)
    #         # print(source_vocab.to_token(list(enc_inputs[0])))
    #         # print(target_vocab.to_token(list(dec_result)))
    #     print(dec_logit)
    #     print('input:', ' '.join(enc_tokens))  
    #     print('pred:', ' '.join(pred_tokens))

        
    #     enc_tokens, dec_tokens = process(sentence)
    #     next_tokens = start_tokens
    #     pred_tokens = []
    #     idx = 0
        
    #     enc_inputs = torch.tensor(source_vocab[enc_tokens], dtype=torch.long, device=device).reshape((1, -1))
    #     while next_tokens != end_tokens:
    #         dec_tokens[idx] = next_tokens
    #         dec_inputs = torch.tensor([target_vocab[dec_tokens]], dtype=torch.long, device=device).reshape((1, -1))
    #         dec_outputs = model_trace(enc_inputs, dec_inputs)
    #         dec_logit = dec_outputs.squeeze(0)
    #         dec_result = dec_logit.argmax(dim=-1)
    #         next_idx = dec_result[idx]
    #         next_tokens = target_vocab.to_token(next_idx)
    #         pred_tokens.append(next_tokens)
    #         idx += 1
    #         if idx >= max_len:
    #             break
    #         # print(dec_outputs.shape)
    #         # print(dec_logit.shape)
    #         # print(dec_result.shape)
    #         # print(source_vocab.to_token(list(enc_inputs[0])))
    #         # print(target_vocab.to_token(list(dec_result)))
    #     print(dec_logit)
    #     print('input:', ' '.join(enc_tokens))  
    #     print('pred:', ' '.join(pred_tokens))
