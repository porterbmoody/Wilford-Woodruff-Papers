#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;



std::vector<std::vector<std::string>> readCSV(const std::string& filename)
{
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);

    if (!file.is_open())
    {
        std::cout << "Failed to open file: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::vector<std::string> row;
        std::stringstream ss(line);
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            row.push_back(cell);
        }

        data.push_back(row);
    }

    file.close();
    return data;
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

std::vector<std::string> tokenize(std::string const &str, const char delim)
{
    std::vector<std::string> tokenized_vector;
    // construct a stream from the string
    std::stringstream ss(str);

    std::string s;
    while (std::getline(ss, s, delim)) {
        tokenized_vector.push_back(s);
    }
    return tokenized_vector;
}

// void find_matches(const std::vector<std::string>& words_woodruff, const std::vector<std::string> words_scripture)
// {
//     std::cout << "finding matches...\n";
//     for (const auto& row : data)
//     {
//         std::vector<std::string> woodruff_words;

//         // for (const std::string& word : woodruff_words) {
//         //     std::cout << word << " ";
//         // }
//         std::string text_woodruff = row[0];
//         std::string text_scripture = row[1];

//         std::vector<std::string> words_woodruff = tokenize(text_woodruff, ',');
//         // std::vector<std::string> words_scripture;
//         // std::cout << get_string('taco');
//     }
// }


// Driver code
int main()
{
    // read in data
    std::string filename = "woodruff_full_strings.csv";
    std::vector<std::vector<std::string>> data = readCSV(filename);

    // text_woodruff, text_scripture = readCSV(filename, data);
    // std::words_woodruff = tokenize()
    // std::vector<std::string> words_woodruff = tokenize();
    // for (const auto& row : data)
    std::cout << "number of total rows: " << data.size() << "\n";
    std::vector<std::string> swag_awesomeness = tokenize("i like pizza");
    std::cout << swag_awesomeness;

    // iterate through each row of dataframe
    // for (int row_number = 0; row_number < data.size(); ++row_number)
    // {
    //     if (row_number == 0) {
    //         continue;
    //     }
    //     std::vector row = data[row_number];
    //     std::cout << "row number: " << row_number << "\n";

    //     // iterate through each cell of current row
    //     for (int column_number = 0; column_number < row.size(); ++column_number)
    //     {
    //         // std::cout << 'col number: ' << col_number;
    //         // current cell
    //         std::string cell = row[column_number];
    //         std::vector<string> cell_tokens = tokenize(cell, ',');
    //         // std::cout << cell << "\n";
    //         // std::cout << cell_tokens;
    //         // display tokenized string of words
    //         std::cout << "number of tokens: " << cell_tokens.size();
    //         for (int i = 0; i < cell_tokens.size(); i++)
    //         {
    //             std::cout << i << "\n";
    //             std::cout << cell_tokens[i] << "\n\n";
    //         }
    //     }
    // }

    return 0;
}
