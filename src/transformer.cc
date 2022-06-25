#include "transformer.h"
#include "vocab.h"
#include "string_utils.h"

#include <torch/script.h>
#include <iostream>


#define DEBUG

Transformer::Transformer(const std::string& model_path, const int& max_len, const std::string& device, Vocab* source_vocab, Vocab* target_vocab) :
    _model(load_model(model_path)),
    _device(device == "cuda" ? torch::kCUDA : torch::kCPU),
    _max_len(max_len),
    _source_vocab(source_vocab),
    _target_vocab(target_vocab),
    _pad(_source_vocab->pad()),
    _bos(_source_vocab->bos()),
    _eos(_source_vocab->eos()),
    _unk(_source_vocab->unk()) {

    _model.to(_device);
    _model.eval();
}

torch::jit::script::Module Transformer::load_model(const std::string& model_path) {
    return torch::jit::load(model_path);
}

torch::Tensor Transformer::get_attn_self_mask(const torch::Tensor& seq_q, \
                                        const torch::Tensor& seq_k, const int& pad_idx) {
    // batch_size x seq_q x seq_k
    // batch_size = 1, seq_q = seq_k = max_len
    int batch_size = seq_q.size(0);
    int seq_q_len = seq_q.size(1);
    int seq_k_len = seq_k.size(1);
    torch::Tensor mask = seq_k.eq(pad_idx).unsqueeze(1).expand({batch_size, seq_q_len, seq_k_len});
    
    return mask.to(torch::kBool).to(_device);
}
torch::Tensor Transformer::get_attn_subsequent_mask(const torch::Tensor& seq) {
    int batch_size = seq.size(0);
    int seq_len = seq.size(1);
    torch::Tensor subsequent_mask = torch::triu(torch::ones({seq_len, seq_len}, torch::kInt32), 1);
    subsequent_mask = subsequent_mask.unsqueeze(0).expand({batch_size, -1, -1});
    return subsequent_mask.to(torch::kBool).to(_device);
}

std::string Transformer::preprocess(const std::string& sentence) {
    std::unordered_set<char> punctuations = {'.', ',', '!', '?'};
    std::string ans;
    for (int i = 0; i < sentence.size(); i++) {
        if (i > 0 and sentence[i-1] != ' ' and punctuations.find(sentence[i]) != punctuations.end()) {
            ans += ' ';
        }
        ans += sentence[i];
    }
    return ans;
}

std::vector<std::string> Transformer::sentence2tokens(const std::string& sentence) {
    std::vector<std::string> tokens = stringUtil::split(sentence, ' ');
    for (auto& token : tokens) {
        token = stringUtil::strip(token);
    }
    return tokens;
}

std::vector<int> Transformer::tokens2ids(const std::vector<std::string>& tokens, Vocab* vocab) {
    if (vocab == nullptr) {
        return {};
    }
    return vocab->get_idx(tokens);
}

std::vector<std::string> Transformer::ids2tokens(const std::vector<int>& ids, Vocab* vocab) {
    if (vocab == nullptr) {
        return {};
    }
    return vocab->get_token(ids);
}

std::string Transformer::tokens2sentence(const std::vector<std::string>& tokens) {
    std::string ans;
    for (int i = 0; i < tokens.size(); i++) {
        ans += tokens[i] + " ";
    }
    ans = stringUtil::strip(ans);
    return ans;
}

std::string Transformer::inference(const std::string& sentence) {
    const int& pad_idx = _source_vocab->get_idx(_pad);
    const int& bos_idx = _source_vocab->get_idx(_bos);
    const int& eos_idx = _source_vocab->get_idx(_eos);
    const int& unk_idx = _source_vocab->get_idx(_unk);
    

    std::vector<std::string> tokens = sentence2tokens(preprocess(sentence));
    std::vector<int> enc_ids = tokens2ids(tokens, _source_vocab);
    std::vector<int> dec_ids = {_target_vocab->get_idx(_bos)};

    // padding
    int enc_len = enc_ids.size();
    for (int i = 0; i < _max_len - enc_len; i++) {
        enc_ids.push_back(pad_idx);
    }
    int dec_len = dec_ids.size();
    for (int i = 0; i < _max_len - dec_len; i++) {
        dec_ids.push_back(pad_idx);
    }

    // encoder inputs and encoder masks remain the same
    torch::Tensor enc_ids_tensor = torch::from_blob(enc_ids.data(), {1, _max_len}, torch::kInt32).to(_device);
    torch::Tensor enc_self_mask = get_attn_self_mask(enc_ids_tensor, enc_ids_tensor, pad_idx);
    
    std::vector<std::string> inference_tokens;
    std::string next_tokens;
    for (int i = 0; i < _max_len-1 and next_tokens != _eos; ++i) {
        torch::Tensor dec_ids_tensor = torch::from_blob(dec_ids.data(), {1, _max_len}, torch::kInt32).to(_device);
        torch::Tensor dec_self_mask = get_attn_self_mask(dec_ids_tensor, dec_ids_tensor, pad_idx);
        torch::Tensor dec_subsquence_mask = get_attn_subsequent_mask(dec_ids_tensor);
        dec_self_mask = dec_self_mask & dec_subsquence_mask;
        torch::Tensor dec_enc_mask = get_attn_self_mask(dec_ids_tensor, enc_ids_tensor, pad_idx);
        torch::Tensor output = _model.forward({enc_ids_tensor, dec_ids_tensor, enc_self_mask, dec_self_mask, dec_enc_mask}).toTensor();
        output = output.squeeze(0);
        output = output.argmax(-1);
        int next_id = output[i].item().toInt();
        // update decoder input
        dec_ids[i+1] = next_id;
        next_tokens = _target_vocab->get_token(next_id);
        // #ifdef DEBUG
        // std::vector<int> tks;
        // for (int i = 0; i < _max_len; ++i) {
        //     tks.push_back(output[i].item().toInt());
        // }
        // std::cout << tokens2sentence(ids2tokens(tks, _target_vocab)) << std::endl;
        // #endif
        inference_tokens.push_back(_target_vocab->get_token(next_id));
    }
    if (inference_tokens.size() > 0) {
        inference_tokens.pop_back();
    }
    return tokens2sentence(inference_tokens);  
}