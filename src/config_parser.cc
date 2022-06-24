#include "string_utils.h"
#include "config_parser.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>


namespace
{
    bool isComment(const std::string& line)
    {
        return line[0] == ';';
    }

    std::string extractSectionName(const std::string& line)
    {
        return std::string(line.begin() + 1, line.end() - 1);
    }

    bool isSectionHeading(const std::string& line)
    {
        if (line[0] != '[' || line[line.size() - 1] != ']')
        {
            return false;
        }
        const std::string sectionName = extractSectionName(line);
        return std::all_of(sectionName.begin(), sectionName.end(), [](char c) { return std::isalpha(c); });
    }

    bool isKeyValuePair(const std::string& line)
    {
        // Assume we have already checked if it's a comment or section header.
        return std::count(line.begin(), line.end(), '=') == 1;
    }

    void ensureSectionIsUnique(const std::string& sectionName, const Config& sections)
    {
        if (sections.count(sectionName) != 0)
        {
            throw std::runtime_error(sectionName + " appears twice in config file");
        }
    }

    void ensureKeyIsUnique(const std::string& key, const Section& section)
    {
        if (section.count(key) != 0)
        {
            throw std::runtime_error(key + " appears twice in section");
        }
    }

    void ensureCurrentSection(const std::string& line, const std::string& currentSection)
    {
        if (currentSection.empty())
        {
            throw std::runtime_error(line + " does not occur within a section");
        }
    }

    std::pair<std::string, std::string> parseKeyValuePair(const std::string& line)
    {
        std::vector<std::string> pair = stringUtil::split(line, '=');
        return std::pair <std::string, std::string>(stringUtil::strip(pair[0]), stringUtil::strip(pair[1]));
    }

}

ConfigParser::ConfigParser(const std::string& pathname) :
    mCurrentSection(""),
    mSections(parseFile(pathname))
{
}

std::vector<std::string> ConfigParser::getSections() const
{
    std::vector<std::string> sectionNames;
    for (auto it = mSections.begin(); it != mSections.end(); ++it)
    {
        sectionNames.push_back(it->first);
    }
    return sectionNames;
}

Config ConfigParser::parseFile(const std::string& pathname)
{
    Config sections;
    std::ifstream input(pathname);
    std::string line;
    while (std::getline(input, line))
    {
        parseLine(line, sections);
    }
    return sections;
}

void ConfigParser::parseLine(const std::string& line, Config& sections)
{
    if (isComment(line))
    {
        return;
    }
    else if (isSectionHeading(line))
    {
        addSection(line, sections);
    }
    else if (isKeyValuePair(line))
    {
        addKeyValuePair(line, sections);
    }
}

void ConfigParser::addSection(const std::string& line, Config& sections)
{
    const std::string sectionName = extractSectionName(line); 
    ensureSectionIsUnique(sectionName, sections);
    sections.insert(std::pair<std::string, Section>(sectionName, Section()));
    mCurrentSection = sectionName;
}

void ConfigParser::addKeyValuePair(const std::string& line, Config& sections) const
{
    ensureCurrentSection(line, mCurrentSection);
    const auto keyValuePair = parseKeyValuePair(line);
    ensureKeyIsUnique(keyValuePair.first, sections.at(mCurrentSection));
    sections.at(mCurrentSection).insert(keyValuePair);
}