#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;



void readCSV(const std::string& filename, std::vector<std::vector<std::string>>& data)
{
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line))
    {
        std::vector<std::string> row;
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ','))
        {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();
}

void printCSV(const std::vector<std::vector<std::string>>& data)
{
    for (const auto& row : data)
    {
        for (const auto& cell : row)
        {
            std::cout << cell << "\t";
        }
        std::cout << std::endl;
    }
}

void tokenize(std::string const &str, const char delim,
            std::vector<std::string> &out)
{
    // construct a stream from the string
    std::stringstream ss(str);

    std::string s;
    while (std::getline(ss, s, delim)) {
        out.push_back(s);
    }
}


void find_matches(const std::vector<std::vector<std::string>>& data)
{
    std::cout << "finding matches...\n";
    for (const auto& row : data)
    {
        std::vector<std::string> woodruff_words;

        // Add strings to the vector
        woodruff_words.push_back("Hello");
        woodruff_words.push_back("World");

        for (const std::string& word : woodruff_words) {
            std::cout << word << " ";
        }
        std::cout << "\nwoodruff text...\n";
        std::cout << row[0] << "\n\n";
        std::cout << "scripture text...\n";
        std::cout << row[1];
        for (const auto& cell : row)
        {
            std::cout << cell << "\n";
        }
    }
}


// Driver code
int main()
{
    std::vector<std::vector<std::string>> data;
    std::string filename = "test1.csv";

    readCSV(filename, data);
    // printCSV(data);
    find_matches(data);

    return 0;
}
