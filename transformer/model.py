import torch
import torch.nn as nn
import numpy as np

import os

class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False):
        # query_size, key_size, value_size = num_hiddens
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = num_hiddens // num_heads
        # self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_heads*self.head_dim, bias=bias)
        self.W_k = nn.Linear(key_size, num_heads*self.head_dim, bias=bias)
        self.W_v = nn.Linear(value_size, num_heads*self.head_dim, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_heads*self.head_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def split_head(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
        return torch.transpose(x, 1, 2) # (batch, heads, seq_len, head_dim)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        dk = k.shape[-1]
        scores = torch.einsum('bnqd,bnkd->bnqk', q, k) / np.sqrt(dk)
        if mask is not None:
            # [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            # mask = mask.eq(1)
            scores.masked_fill_(mask, -1e9)
            
        attention_weight = torch.softmax(scores, dim=-1)
        context = torch.einsum('bnqk,bnkd->bnqd', attention_weight, v)
        context = context.permute(0, 2, 1, 3)
        return context.reshape(context.shape[0], context.shape[1], -1)
    
        
    def forward(self, queries, keys, values, mask=None):
        # shape of queries，keys，values: (batch_size，num_kv_pairs，num_hiddens)
        q = self.W_q(queries)
        k, v = self.W_k(keys), self.W_v(values)
        _q = self.split_head(q)
        _k, _v = self.split_head(k), self.split_head(v)
        context = self.scaled_dot_product_attention(_q, _k, _v, mask)
        o = self.W_o(context)
        o = self.dropout(o)
        return o    # (batch_size, seq_len, num_hiddens)
    
 
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000) -> None:
        super(PositionEncoding, self).__init__()
        self.pe = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32)/num_hiddens)
        # x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, 2*torch.arange(num_hiddens, dtype=torch.float32)/num_hiddens)
        self.pe[:, :, 0::2] = torch.sin(x)
        self.pe[:, :, 1::2] = torch.cos(x)
        
    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].to(x.device)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, 
                 num_hiddens, ffn_num_hiddens, 
                 num_heads, dropout, use_bias=False) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(query_size, key_size, value_size, num_hiddens,
                                            num_heads, dropout, use_bias)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.layernorm1 = nn.LayerNorm(num_hiddens)
        self.layernorm2 = nn.LayerNorm(num_hiddens)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, self_attn_mask=None):
        att = self.attention(x, x, x, self_attn_mask)
        o1 = self.layernorm1(att + x)
        ffn = self.dropout(self.ffn(o1))
        o = self.layernorm2(ffn + o1)
        return o
        

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, 
                 query_size, key_size, value_size, 
                 num_hiddens, ffn_num_hiddens, 
                 num_heads, num_layers, 
                 dropout, use_bias=False) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, max_len)
        self.encoder_layers = nn.Sequential()
        for i in range(num_layers):
            self.encoder_layers.add_module(f'encoder_layer_{i}',
                                           EncoderLayer(query_size=query_size, 
                                                        key_size=key_size, 
                                                        value_size=value_size, 
                                                        num_hiddens=num_hiddens, 
                                                        ffn_num_hiddens=ffn_num_hiddens, 
                                                        num_heads=num_heads, 
                                                        dropout=dropout, 
                                                        use_bias=use_bias)
                                           )
        
    def forward(self, x, self_attn_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, self_attn_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, query_size, key_size, value_size, 
                 num_hiddens, ffn_num_hiddens, 
                 num_heads, dropout, use_bias=False) -> None:
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.enc_dec_attn = MultiHeadAttention(query_size, key_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.layernorm1 = nn.LayerNorm(num_hiddens)
        self.layernorm2 = nn.LayerNorm(num_hiddens)
        self.layernorm3 = nn.LayerNorm(num_hiddens)
        self.ffn = PositionWiseFFN(num_hiddens, ffn_num_hiddens, num_hiddens)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_dec_attn_mask):
        # mask.shape = (batch, seq_len, seq_len)
        dec_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        dec_o1 = self.layernorm1(dec_attn + dec_inputs)
        enc_dec_attn = self.enc_dec_attn(dec_o1, enc_outputs, enc_outputs, enc_dec_attn_mask)
        dec_o2 = self.layernorm2(enc_dec_attn + dec_o1)
        ffn = self.dropout(self.ffn(dec_o2))
        o = self.layernorm3(ffn + dec_o2)
        return o
        

class Decoder(nn.Module):
    def __init__(self, vocab_size, max_len, 
                 query_size, key_size, value_size, 
                 num_hiddens, ffn_num_hiddens, 
                 num_heads, num_layers, dropout, use_bias=False) -> None:
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, max_len)
        self.decoder_layers = nn.Sequential()
        for i in range(num_layers):
            self.decoder_layers.add_module(f'decoder_layer_{i}',
                                           DecoderLayer(query_size=query_size, 
                                                        key_size=key_size, 
                                                        value_size=value_size, 
                                                        num_hiddens=num_hiddens, 
                                                        ffn_num_hiddens=ffn_num_hiddens, 
                                                        num_heads=num_heads, 
                                                        dropout=dropout, 
                                                        use_bias=use_bias)
                                           )

    def forward(self, dec_inputs, enc_outputs, self_attn_mask, enc_dec_attn_mask):
        dec_inputs_emd = self.embedding(dec_inputs)
        dec_inputs_emd = self.pos_encoding(dec_inputs_emd)
        for layer in self.decoder_layers:
            o = layer(dec_inputs_emd, enc_outputs, self_attn_mask, enc_dec_attn_mask)
        return o
        
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, max_len, pad_idx,
                 query_size, key_size, value_size, 
                 num_hiddens, ffn_num_hiddens, 
                 num_heads, enc_layers, dec_layers, dropout, use_bias=False) -> None:
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(vocab_size=src_vocab,
                               max_len=max_len,
                               query_size=query_size,
                               key_size=key_size,
                               value_size=value_size,
                               num_hiddens=num_hiddens,
                               ffn_num_hiddens=ffn_num_hiddens,
                               num_heads=num_heads,
                               num_layers=enc_layers,
                               dropout=dropout,
                               use_bias=use_bias)
        self.decoder = Decoder(vocab_size=tgt_vocab,
                               max_len=max_len,
                               query_size=query_size,
                               key_size=key_size,
                               value_size=value_size,
                               num_hiddens=num_hiddens,
                               ffn_num_hiddens=ffn_num_hiddens,
                               num_heads=num_heads,
                               num_layers=dec_layers,
                               dropout=dropout,
                               use_bias=use_bias)
        self.projection = nn.Linear(num_hiddens, tgt_vocab)
        
    def forward(self, enc_inputs, dec_inputs):
        enc_self_attn_mask = self._attn_pad_mask(enc_inputs, enc_inputs)
        encoder_outputs = self.encoder(enc_inputs, enc_self_attn_mask)
        dec_self_attn_mask = self._attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_subsequent_attn_mask = self._attn_subsequent_mask(dec_inputs)
        dec_self_attn_mask = dec_self_attn_mask + dec_self_subsequent_attn_mask
        # 尽量不使用替换操作, 建议改为`dec_self_attn_mask = dec_self_attn_mask.gt(0)`
        # dec_self_attn_mask[:] = dec_self_attn_mask.gt(0)
        dec_self_attn_mask = dec_self_attn_mask.gt(0)
        
        dec_enc_attn_mask = self._attn_pad_mask(dec_inputs, enc_inputs)
        decoder_outputs = self.decoder(dec_inputs, encoder_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        dec_logits = self.projection(decoder_outputs)
        return dec_logits
    
    def _attn_pad_mask(self, seq_q, seq_k):
        batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
        batch_size, len_k = seq_k.size()
        # `seq_k.data`会导致tracing之后的模型预测结果不符合预期，需要把`seq_k.data`改成`seq_k`
        # 参考https://pytorch.org/docs/stable/onnx.html#avoid-tensor-data
        # pad_mask = seq_k.data.eq(self.pad_idx).unsqueeze(1).expand(batch_size, len_q, len_k)
        pad_mask = seq_k.eq(self.pad_idx).unsqueeze(1).expand(batch_size, len_q, len_k)
        return pad_mask.to(seq_q.device) # [batch_size, len_q, len_k]
        
    def _attn_subsequent_mask(self, seq):
        batch_size, len_seq = seq.size()
        # 生成一个上三角矩阵
        subsequent_mask = torch.triu(torch.ones((len_seq, len_seq), dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
        return subsequent_mask.to(seq.device)
    

class TransformerWithoutMask(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, max_len, pad_idx,
                 query_size, key_size, value_size, 
                 num_hiddens, ffn_num_hiddens, 
                 num_heads, enc_layers, dec_layers, dropout, use_bias=False) -> None:
        super(TransformerWithoutMask, self).__init__()
        self.pad_idx = pad_idx
        self.encoder = Encoder(vocab_size=src_vocab,
                               max_len=max_len,
                               query_size=query_size,
                               key_size=key_size,
                               value_size=value_size,
                               num_hiddens=num_hiddens,
                               ffn_num_hiddens=ffn_num_hiddens,
                               num_heads=num_heads,
                               num_layers=enc_layers,
                               dropout=dropout,
                               use_bias=use_bias)
        self.decoder = Decoder(vocab_size=tgt_vocab,
                               max_len=max_len,
                               query_size=query_size,
                               key_size=key_size,
                               value_size=value_size,
                               num_hiddens=num_hiddens,
                               ffn_num_hiddens=ffn_num_hiddens,
                               num_heads=num_heads,
                               num_layers=dec_layers,
                               dropout=dropout,
                               use_bias=use_bias)
        self.projection = nn.Linear(num_hiddens, tgt_vocab)
        
    def forward(self, enc_inputs, dec_inputs, enc_self_attn_mask, dec_self_attn_mask, dec_enc_attn_mask):
        encoder_outputs = self.encoder(enc_inputs, enc_self_attn_mask)
        decoder_outputs = self.decoder(dec_inputs, encoder_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        dec_logits = self.projection(decoder_outputs)
        return dec_logits
    



if __name__ == '__main__':
    query_size, key_size, value_size = 12, 12, 12
    num_hiddens = 12
    ffn_num_hiddens = 24
    num_heads = 2
    encoder_layers = 1
    decoder_layers = 1
    batch_size = 1
    max_len = 20
    source_vocab = 10
    target_vocab = 20
    
    dropout = 0.5
    use_bias = False
    
    model = Transformer(src_vocab=source_vocab,
                        tgt_vocab=target_vocab,
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
    X = torch.ones((batch_size, 4), dtype=torch.long)
    Y = torch.ones((batch_size, 4), dtype=torch.long)
    print(model.eval())
    print(model(X, Y))
    # num_hiddens, num_heads = 100, 5
    
    
    # batch_size, num_queries = 2, 4
    # num_kvpairs = 6
    # attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
    #                             num_hiddens, num_heads, 0.5)
    # print(attention.eval())
    # X = torch.ones((batch_size, num_queries, num_hiddens))
    # Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
    # seq_mask = torch.zeros(batch_size, num_queries, num_kvpairs)
    # print(attention(X, Y, Y, seq_mask).shape)
    
    # num_hiddens = 24
    # num_heads = 3
    # batch_size = 2
    
    # seq_len = 100
    # X = torch.ones((batch_size, seq_len), dtype=torch.int32)
    # pad_mask = torch.zeros(batch_size, seq_len, seq_len)
    
    # # encoder_blk = EncoderLayer(24,24,24,num_hiddens,num_heads, 48, 0.5)
    # # print(encoder_blk.eval())
    # # print(encoder_blk(X, pad_mask).shape)
    # encoder = Encoder(10000, seq_len, 24,24,24,num_hiddens,48, num_heads, 2, 0.5)
    # print(encoder.eval())
    # print(encoder(X, pad_mask).shape)
