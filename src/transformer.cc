#include "transformer.h"
#include "vocab.h"
#include "string_utils.h"

#include <torch/script.h>


Transformer(const std::string& model_path, const int& max_len, Vocab* source_vocab, Vocab* target_vocab) :
    _model(load_model(model_path)),
    _max_len(max_len),
    _source_vocab(source_vocab),
    _target_vocab(target_vocab) {

}

torch::jit::script::Module Transformer::load_model(const std::string& model_path) {
    return torch::jit::load(model_path);
}

torch::Tensor Transformer::get_attn_self_mask(const vector<std::string>& seq_q, \
                                        const vector<std::string>& seq_k, const int& pad_idx) {

}
torch::Tensor Transformer::get_attn_subsequent_mask(const vector<std::string>& seq) {

}

vector<std::string> Transformer::sentence2tokens(const std::string& sentence) {

}

vector<int> Transformer::tokens2ids(const vector<std::string>& tokens, Vocab* vocab) {
    if (vocab == nullptr) {
        return {};
    }
    return vocab->get_idx(tokens);
}

vector<std::string> Transformer::ids2tokens(const vector<int>& ids, Vocab* vocab) {
    if (vocab == nullptr) {
        return {};
    }
    return vocab->get_token(ids);
}

std::string Transformer::tokens2sentence(cconst vector<std::string>& tokens) {

}

std::string Transformer::inference(const std::string& sentence) {

}