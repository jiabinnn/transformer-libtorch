#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "model_tensorrt.h"
#include "config_parser.h"
#include "vocab.h"

int main() {
    // const std::string onnxfile("../testmodel.onnx");
    // Logger logger = Logger();
    // compile("fp16", 1, onnxfile, "../testmodel.engine", logger);
    // std::shared_ptr<nvinfer1::ICudaEngine> engine = load("../testmodel.engine", logger);
    // inference(engine.get(), logger);

    // parser ini file
    std::string ini_file = "config/config.ini";
    ConfigParser parser(ini_file);

    std::string batch_size = parser.get("trainer", "batch_size");
    std::string source_vocab_path = parser.get("data", "source_vocab_path");
    std::string target_vocab_path = parser.get("data", "target_vocab_path");
    std::string device = parser.get("trainer", "device");
    std::string model1_onnx_path = parser.get("save", "model2_onnx_path");
    std::string model1_engine_path = parser.get("save", "model2_engine_path");
    int max_len = atoi(parser.get("trainer", "max_len").c_str());

    std::unique_ptr<Vocab> src_vocab(new Vocab(source_vocab_path));
    std::unique_ptr<Vocab> tgt_vocab(new Vocab(target_vocab_path));

    std::cout << "batch_size=" << batch_size << std::endl;
    std::cout << "source_vocab_path=" << source_vocab_path << std::endl;
    std::cout << "target_vocab_path=" << target_vocab_path << std::endl;
    std::cout << "model_onnx_path=" << model1_onnx_path << std::endl;
    std::cout << "model_engine_path=" << model1_engine_path << std::endl;
    std::cout << "device=" << device << std::endl;
    std::cout << "max_len=" << max_len << std::endl;
    
    std::cout << "src_vocab size=" << src_vocab->get_vocab_size() << std::endl;
    std::cout << "tgt_vocab size=" << tgt_vocab->get_vocab_size() << std::endl;
    
    Engine* engine = new Engine(model1_onnx_path, max_len, device, src_vocab.get(), tgt_vocab.get());
    engine->compile("fp16", 1, model1_onnx_path, model1_engine_path);
    engine->load_network(model1_engine_path);
    
    std::string source_sentence;

    std::cout << "input sentence: ";
    while (getline(std::cin, source_sentence)) {
        if (source_sentence == "q") {
            std::cout << "quit" << std::endl;
            break;
        }
        std::cout << "input: " << source_sentence << std::endl;
        std::string target_sentence = engine->inference(source_sentence);
        std::cout << "pred: " << target_sentence << std::endl;
        std::cout << "input sentence: ";
    }
    return 0;
}