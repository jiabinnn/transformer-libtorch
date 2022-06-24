import torch
import torch.nn as nn
import numpy as np
from model import Transformer
import data_eng2fra

import os


if __name__ == '__main__':
    batch_size = 64
    max_len = 10

    corpus = data_eng2fra.Corpus('data/fra-eng/fra.txt')
    train_iter, source_vocab, target_vocab = corpus.get_dataset(batch_size=batch_size, num_steps=max_len, num_examples=600)
    
    query_size, key_size, value_size = 32,32,32
    num_hiddens = 32
    ffn_num_hiddens = 64
    num_heads = 4
    encoder_layers = 4
    decoder_layers = 4
    source_vocab_size = len(source_vocab)
    target_vocab_size = len(target_vocab)
    
    dropout = 0.1
    use_bias = False
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Transformer(src_vocab=source_vocab_size,
                        tgt_vocab=target_vocab_size,
                        max_len=max_len,
                        pad_idx=1,
                        query_size=query_size,
                        key_size=key_size,
                        value_size=value_size,
                        num_hiddens=num_hiddens,
                        ffn_num_hiddens=ffn_num_hiddens,
                        num_heads=num_heads,
                        enc_layers=encoder_layers,
                        dec_layers=decoder_layers,
                        dropout=dropout,
                        use_bias=use_bias)
    x = torch.ones((batch_size, max_len), dtype=torch.long)
    y = torch.ones((batch_size, max_len), dtype=torch.long)
    