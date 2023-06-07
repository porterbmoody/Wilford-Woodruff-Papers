import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist


class DataUtility:
    """utility class for cleaning the data
    """

    @staticmethod
    def split_string(string):
        words = string.split(' ')
        return words

    @staticmethod
    def combine_rows(column):
        """ combines the top num_rows rows into 1 single string
        """
        return ' '.join(str(cell) for cell in column)

    @staticmethod
    def create_frequency_distribution(string):
        string = string.lower()
        tokens = word_tokenize(string)
        frequency_distribution = FreqDist(tokens)
        data = pd.DataFrame(list(frequency_distribution.items()), columns=['word', 'frequency'])
        print(data.shape)
        return data.sort_values(by = 'frequency', ascending = False)

    @staticmethod
    def remove_new_lines(string):
        return re.sub(r'\n', r' ', string)

    @staticmethod
    def regex_filter(dataframe, column, regex):
        return dataframe[dataframe[column].str.contains(regex) == False]

    @staticmethod
    def replace_regex(string, regex, replacement = None):
        if replacement is None:
            string = re.sub(regex, r'\1', string)
            return string
        string = re.sub(regex, replacement, string)
        return string

    @staticmethod
    def split_string_into_list(text, n):
        words = text.split()
        result = []
        for i in range(0, len(words), n):
            if i + n < len(words) + round(n/2):
                phrase = ' '.join(words[i : i + n])
                result.append(phrase)
            else:
                phrase = ' '.join(words[i : i + n])
                result[-1] += ' ' + phrase
        return result