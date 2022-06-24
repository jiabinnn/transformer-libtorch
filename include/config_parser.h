#pragma once
#include <map>
#include <string>
#include <vector>

typedef std::map<std::string, std::string> Section;
typedef std::map<std::string, Section> Config;

class ConfigParser
{
public:
    ConfigParser(const std::string& pathname);
    Section getSection(const std::string& sectionName) const
    {
        if (mSections.find(sectionName) == mSections.end())
        {
            return std::map<std::string, std::string>();
        }
        return mSections.at(sectionName);
    }
    std::string get(const std::string& sectionName, const std::string& key) const
    {   if (mSections.find(sectionName) == mSections.end() or 
            mSections.at(sectionName).find(key) == mSections.at(sectionName).end())
        {
            return "";
        }
        return mSections.at(sectionName).at(key);
    }
    std::vector<std::string> getSections() const;

private:
    Config parseFile(const std::string& pathname);
    void parseLine(const std::string& line, Config& sections);
    void addSection(const std::string& line, Config& sections);
    void addKeyValuePair(const std::string& line, Config& sections) const;

    std::string mCurrentSection;
    const Config mSections;
};