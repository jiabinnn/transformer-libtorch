#pragma once

#include <string>
#include <vector>
#include <memory>

#include <NvInfer.h>
#include <buffers.h>

class Vocab;


class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};


class Engine {
public:
    Engine(const std::string& model_path, const int& max_len, const std::string& device, Vocab* source_vocab, Vocab* target_vocab);
    bool load_model(const std::string& model_path);
    
    std::string preprocess(const std::string& sentence);
    std::vector<std::string> sentence2tokens(const std::string& sentence);
    std::vector<int> tokens2ids(const std::vector<std::string>& tokens, const std::unique_ptr<Vocab>& vocab);

    std::vector<std::string> ids2tokens(const std::vector<int>& ids, const std::unique_ptr<Vocab>& vocab);
    std::string tokens2sentence(const std::vector<std::string>& tokens);
    bool compile(const std::string& mode, const unsigned int& max_batchsize, 
            const std::string& onnx_file, const std::string& engine_file);
    bool load_network(const std::string& enginefile);
    std::string inference(const std::string& sentence);
private:
    std::unique_ptr<nvinfer1::ICudaEngine> _engine;
    std::unique_ptr<nvinfer1::IExecutionContext> _context;
    Logger _logger;
    cudaStream_t _stream;

    samplesCommon::ManagedBuffer _inputBuff;
    samplesCommon::ManagedBuffer _outputBuff;

    int _max_len;
    std::unique_ptr<Vocab> _source_vocab;
    std::unique_ptr<Vocab> _target_vocab;
    std::string _pad;
    std::string _bos;
    std::string _eos;
    std::string _unk;
};
