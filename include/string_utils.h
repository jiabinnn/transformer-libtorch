#pragma once
#include <string>
#include <vector>

// The function in this namespace will go into a custom string util library
namespace stringUtil
{
    std::vector<std::string> split(const std::string& string, char delimiter = ' ');
    
    std::string strip(const std::string& str);
}
