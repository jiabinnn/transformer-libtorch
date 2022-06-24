#pragma once

#include <string>
#include <vector>
#include <map>

class Vocab {
public:
    Vocab(std::string path);
    std::string get_token(const int& idx);

    std::vector<std::string> get_token(const std::vector<int>& indice);

    int get_idx(const std::string& token);
    
    std::vector<int> get_idx(const std::vector<std::string>& tokens);

    std::string unk() { return "<unk>"; }
    std::string bos() { return "<bos>"; }
    std::string eos() { return "<eos>"; }
    std::string pad() { return "<pad>"; }

    int get_vocab_size();
private:
    std::vector<std::string> idx2token;
    std::map<std::string, int> token2idx;

    void parse_line(std::string line);
};