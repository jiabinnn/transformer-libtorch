#include "string_utils.h"
#include <fstream>
#include <sstream>
// The function in this namespace will go into a custom string util library
namespace stringUtil
{
    std::vector<std::string> split(const std::string& string, char delimiter)
    {
        std::vector<std::string> result;
        std::stringstream stream(string);
        std::string word;
        while (std::getline(stream, word, delimiter))
        {
            result.push_back(word);
        }
        return result;
    }

    std::string strip(const std::string& str) {
        int l = 0;
        int r = str.size() - 1;
        while (l < str.size() && str[l] == ' ') {
            l++;
        }
        while (r >= 0 && str[r] == ' ') {
            r--;
        }
        return str.substr(l, r - l + 1);
    }
}
