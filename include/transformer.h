#pragma once

#include <string>
#include <vector>
#include <torch/script.h>

class Vocab;

class Transformer {
public:
    Transformer(const std::string& model_path, const int& max_len, const std::string& device, Vocab* source_vocab, Vocab* target_vocab);
    torch::jit::script::Module load_model(const std::string& model_path);
    torch::Tensor get_attn_self_mask(const torch::Tensor& seq_q, const torch::Tensor& seq_k, const int& pad_idx);
    torch::Tensor get_attn_subsequent_mask(const torch::Tensor& seq);

    std::string preprocess(const std::string& sentence);
    std::vector<std::string> sentence2tokens(const std::string& sentence);
    std::vector<int> tokens2ids(const std::vector<std::string>& tokens, Vocab* vocab);

    std::vector<std::string> ids2tokens(const std::vector<int>& ids, Vocab* vocab);
    std::string tokens2sentence(const std::vector<std::string>& tokens);

    std::string inference(const std::string& sentence);
private:
    torch::jit::script::Module _model;
    torch::Device _device;
    int _max_len;
    Vocab* _source_vocab;
    Vocab* _target_vocab;
    std::string _pad;
    std::string _bos;
    std::string _eos;
    std::string _unk;
};
