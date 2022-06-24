#include "config_parser.h"
#include "vocab.h"


#include <string>
#include <vector>
#include <iostream>

int main(int argc, char* argv[])
{
    // load config
    std::string ini_file = "config/config.ini";
    ConfigParser config(ini_file);
    int batch_size = atoi(config.get("trainer", "batch_size").c_str());
    int max_len = atoi(config.get("trainer", "max_len").c_str());
    

    std::cout << "batch_size=" << batch_size << std::endl;

    return 0;
}