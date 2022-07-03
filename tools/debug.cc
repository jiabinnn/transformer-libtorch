#include <string>
#include "config_parser.h"
#include "string_utils.h"
#include "vocab.h"
#include "model_libtorch.h"

#include <vector>
#include <iostream>


int main(int argc, char* argv[])
{
    // parser ini file
    std::string ini_file = "config/config.ini";
    ConfigParser parser(ini_file);
    std::vector<std::string> s = parser.getSections();
    for (auto c: s) {
        std::cout << c << std::endl;
    }
    std::map<std::string, std::string> sec = parser.getSection("trainer");
    for (auto iter = sec.begin(); iter != sec.end(); iter++) {
        std::cout << iter->first << "|" << iter->second << std::endl;
    }
    std::string batch_size = parser.get("trainer", "batch_size");
    std::cout << "batch_size=" << batch_size << std::endl;



    std::string source_vocab_path = parser.get("data", "source_vocab_path");
    std::string target_vocab_path = parser.get("data", "target_vocab_path");
    Vocab src_vocab(source_vocab_path);
    Vocab tgt_vocab(target_vocab_path);

    std::cout << "size=" << src_vocab.get_vocab_size() << std::endl;
    
    std::string model_path = parser.get("save", "model1_trace_path");
    std::string device = parser.get("trainer", "device");
    int max_len = atoi(parser.get("trainer", "max_len").c_str());

    Transformer* model = new Transformer(model_path, max_len, device, &src_vocab, &tgt_vocab);
    
    model->inference("i am busy .");
    
    return 0;
}