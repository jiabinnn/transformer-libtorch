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

    std::string batch_size = parser.get("trainer", "batch_size");
    std::string source_vocab_path = parser.get("data", "source_vocab_path");
    std::string target_vocab_path = parser.get("data", "target_vocab_path");
    std::string model_path = parser.get("save", "model1_trace_path");
    std::string device = parser.get("trainer", "device");
    int max_len = atoi(parser.get("trainer", "max_len").c_str());

    Vocab src_vocab(source_vocab_path);
    Vocab tgt_vocab(target_vocab_path);

    std::cout << "batch_size=" << batch_size << std::endl;
    std::cout << "source_vocab_path=" << source_vocab_path << std::endl;
    std::cout << "target_vocab_path=" << target_vocab_path << std::endl;
    std::cout << "model_path=" << model_path << std::endl;
    std::cout << "device=" << device << std::endl;
    std::cout << "max_len=" << max_len << std::endl;
    
    std::cout << "src_vocab size=" << src_vocab.get_vocab_size() << std::endl;
    std::cout << "tgt_vocab size=" << tgt_vocab.get_vocab_size() << std::endl;
    
    Transformer* model = new Transformer(model_path, max_len, device, &src_vocab, &tgt_vocab);
    
    std::string source_sentence;

    std::cout << "input sentence: ";
    while (getline(std::cin, source_sentence)) {
        if (source_sentence == "q") {
            std::cout << "quit" << std::endl;
            break;
        }
        std::cout << "input: " << source_sentence << std::endl;
        std::string target_sentence = model->inference(source_sentence);
        std::cout << "pred: " << target_sentence << std::endl;
        std::cout << "input sentence: ";
    }
    return 0;
}