#include "vocab.h"
#include "string_utils.h"

#include <fstream>
#include <string>
#include <vector>
#include <map>


Vocab::Vocab(std::string path) {
    std::ifstream input(path);
    std::string line;
    while (std::getline(input, line))
    {
        parse_line(line);
    }
}


std::string Vocab::get_token(const int& idx) {
    if (idx < 0 || idx >= idx2token.size()) {
        return unk();
    }
    return idx2token[idx];
}

std::vector<std::string> Vocab::get_token(const std::vector<int>& indice) {
    std::vector<std::string> ans;
    for (int i = 0; i < indice.size(); ++i) {
        ans.push_back(get_token(indice[i]));
    }
    return ans;
}

int Vocab::get_idx(const std::string& token) {
    if (token2idx.find(token) == token2idx.end()) {
        return token2idx.at(unk());
    }
    return token2idx.at(token);
}

std::vector<int> Vocab::get_idx(const std::vector<std::string>& tokens) {
    std::vector<int> ans;
    for (int i = 0; i < tokens.size(); ++i) {
        ans.push_back(get_idx(tokens[i]));    
    }
    return ans;
}

int Vocab::get_vocab_size() {
    return idx2token.size();
}


void Vocab::parse_line(std::string line) {
    std::vector<std::string> pair = stringUtil::split(line, '\t');
    if (pair.size() != 2) {
        return;
    }

    idx2token.push_back(pair[1]);
    token2idx[pair[1]] = atoi(pair[0].c_str());
}
