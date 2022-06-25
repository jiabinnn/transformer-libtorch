import collections
import logging
import os, sys
import time, datetime
import pickle

import numpy as np
import torch

logging.getLogger().setLevel(logging.INFO)

def get_attn_pad_mask(seq_q, seq_k, pad_idx):
        batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
        batch_size, len_k = seq_k.size()
        pad_mask = seq_k.data.eq(pad_idx).unsqueeze(1).expand(batch_size, len_q, len_k)
        return pad_mask.to(seq_k.device)     # [batch_size, len_q, len_k]

# def get_attn_pad_mask(valid_len, max_len):
#     mask = torch.arange((max_len), dtype=torch.float32)[None, :] >= valid_len[:, None]
#     return mask     # mask[pad_value] = true
        
def get_attn_subsequent_mask(seq):
    batch_size, len_seq = seq.size()
    # 生成一个上三角矩阵
    subsequent_mask = torch.triu(torch.ones((len_seq, len_seq), dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(batch_size, -1, -1)
    return subsequent_mask.to(seq.device)  # [batch_size, seq_len, seq_len]

def sequence_mask(X, valid_len, pad_idx=1):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = pad_idx
    return X

# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# sequence_mask(X, torch.tensor([1, 2]))

class Vocab(object):
    def __init__(self, tokens=None, min_freq=1, reserved_tokens=None, savepath=None, loadpath=None):
        
        self.token2idx = {}
        self.idx2token = []

        if loadpath is not None:
            self.load_vocab(loadpath)
            return 
        
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        self.reserved_tokens = reserved_tokens
            
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx2token = ['<unk>'] + self.reserved_tokens
        self.token2idx = {
            token: idx
            for idx, token in enumerate(self.idx2token)
        }
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token2idx:
                self.idx2token.append(token)
                self.token2idx[token] = len(self.idx2token) - 1
        if savepath is not None:
            self.save_vocab(savepath)
                            
    def save_vocab(self, path):
        with open(path, 'w') as f:
            for idx, token in enumerate(self.idx2token):
                f.write('\t'.join([str(idx), str(token)]) + '\n')
    
    def load_vocab(self, path):    
        with open(path, 'r') as f:
            for line in f.readlines():
                idx, token = line.strip().split('\t')
                self.idx2token.append(token)
                self.token2idx[token] = int(idx)    
    
    def count_corpus(self, tokens):
        if len(tokens) == 0 or isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)
    
    def __len__(self):
        return len(self.idx2token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_token(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx2token[indices]
        return [self.to_token(index) for index in indices]
    
    @property
    def unk(self):
        return 0
    
    # @property
    # def token_freqs(self):
    #     return self._token_freqs
                

class Corpus(object):
    def __init__(self, data_file):
        self.data_file = data_file

    def read_data_file(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def preprocess(self, text):
        def no_space(char, prev_char):
            return char in set(',.!?') and prev_char != ' '
        # 使用空格替换不间断空格
        # 使用小写字母替换大写字母
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # 在单词和标点符号之间插入空格
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
            for i, char in enumerate(text)]
        return ''.join(out)

    def tokenize(self, text, num_examples=None):
        source, target = [], []
        for i, line in enumerate(text.split('\n')):
            if num_examples and i >= num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(parts[1].split(' '))
        return source, target

    def truncate_pad(self, line, num_steps, padding_token):
        if len(line) > num_steps:
            return line[:num_steps] # truncate 
        
        return line + [padding_token] * (num_steps - len(line)) # pad    
    
    def build_array(self, lines, vocab, num_steps):
        lines = [vocab[l] for l in lines]
        lines = [l + [vocab['<eos>']] for l in lines]
        lines = [self.truncate_pad(l, num_steps, vocab['<pad>']) for l in lines]
        array = torch.tensor(lines)
        valid_len = (array != vocab['<pad>']).sum(1)
        return array, valid_len
        
    def get_dataset(self, batch_size, num_steps, num_examples=None, shuffle=True):
        raw_text = self.read_data_file(self.data_file)
        text = self.preprocess(raw_text)
        source, target = self.tokenize(text, num_examples)    
        source_vocab = Vocab(source, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'], savepath='data/fra-eng/source_vocab')
        target_vocab = Vocab(target, min_freq=3, reserved_tokens=['<pad>', '<bos>', '<eos>'], savepath='data/fra-eng/target_vocab')
        source_array, source_valid_len = self.build_array(source, source_vocab, num_steps)
        target_array, target_valid_len = self.build_array(target, target_vocab, num_steps)
        data_arrays = (source_array, source_valid_len, target_array, target_valid_len)
        dataset = torch.utils.data.TensorDataset(*data_arrays)
        train_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_iter, source_vocab, target_vocab
    

if __name__ == '__main__':
    # vocab = Vocab('data/fra-eng/fra.txt')
    # raw_text = vocab.read_data_file()
    # text = vocab.preprocess(raw_text)
    # source, target = vocab.tokenize(text)
    # print(len(source), len(target))

    corpus = Corpus('data/fra-eng/fra.txt')
    train_iter, source_vocab, target_vocab = corpus.get_dataset(batch_size=2, num_steps=10, num_examples=600, shuffle=False)
    # for x, x_len, y, y_len in train_iter:
    #     print(x, source_vocab.to_token(list(x[0])))
    #     print(x_len)
    #     print(y, target_vocab.to_token(list(y[0])))
    #     print(y_len)