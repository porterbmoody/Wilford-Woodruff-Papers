import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist



class DataUtil:
    """ Utility class for cleaning the data
    """

    stop_words = {
        ' as ' : ' ',
        r' is ' : ' ',
        r'\-\-' : ' ',
        ' and ' : ' ',
        r' the ' : ' ',
        ' that ' : ' ',
        ' of ' : ' ',
        ' to ' : ' ',
        ' with ' : ' ',
        ' at ' : ' ',
        ' by ' : ' ',
        ' in ' : ' ',
        ' on ' : ' ',
        ' for ' : ' ',
        ' us ' : ' ',
        ' we ' : ' ',
        ' my ' : ' ',
        ' his ' : ' ',
        ' you ' : ' ',
        r' be ' : ' ',
        r' it ' : ' ',
        r'\.' : '',
        r' i ' : ' ',
        r'\,' : '',
        r'\d+' : ' ',
        r'\:' : '',
        }

    @staticmethod
    def str_split(string, remove_duplicates = True):
        if type(string) == float:
            print(string)
        words = string.split(' ')
        if remove_duplicates:
            words = list(set(words))
        return words

    @staticmethod
    def combine_rows(column):
        """ pass in a column, or list of strings, and it returns a joined string separated by ' '
        """
        return ' '.join(str(cell) for cell in column)

    @staticmethod
    def regex_filter(dataframe, column, regex):
        """ Remove rows that have regex match in certain string column
        """
        return dataframe[dataframe[column].str.contains(regex) == False]

    @staticmethod
    def str_replace(string, regex, replacement):
        return re.sub(regex, replacement, string)

    @staticmethod
    def str_extract(string, regex):
        return re.findall(regex, string)

    @staticmethod
    def str_count_occurrences(string, word):
        return DataUtil.str_extract(string, word)

    @staticmethod
    def str_remove(string, regex):
        return re.sub(regex, r'', string)

    @staticmethod
    def str_detect(string, regex):
        return bool(re.search(pattern = regex, string=string))

    @staticmethod
    def str_remove_list(string, regex_list):
        for regex in regex_list:
            string = re.sub(regex, '', string)
        return string

    @staticmethod
    def str_replace_list(string, regex_list, replacement):
        for regex in regex_list:
            if regex == r'\s\d+\s' and DataUtil.str_detect(string, regex):
                print("PIZZA")
                print(string)
                print(re.sub(regex, replacement, string))
            string = re.sub(regex, replacement, string)
        return string

    @staticmethod
    def remove_stop_words(string):

        return DataUtil.str_replace_list(string, regex_list = stop_words, replacement = ' ')

    @staticmethod
    def split_string_into_list(string, n):
        """ Basically converts a string of text into a list of strings of text
        each element containing n words.
        Except the last few words get attatched on the end of the last element of the list,
        but only if the number of remaining words is > n/2 or more than half of n
        """
        words = string.split()
        result = []
        for i in range(0, len(words), n):
            if i + n < len(words) + round(n/2) or len(result) == 0:
                phrase = ' '.join(words[i : i + n])
                result.append(phrase)
            else:
                phrase = ' '.join(words[i : i + n])
                result[-1] += ' ' + phrase
        return result

    @staticmethod
    def create_frequency_dist(string):
        """ Returns pandas dataframe containing frequencies of each word in string
        """
        string = string.lower()
        tokens = word_tokenize(string)
        frequency_distribution = FreqDist(tokens)
        data = pd.DataFrame(list(frequency_distribution.items()), columns=['word', 'frequency'])
        print(data.shape)
        return data.sort_values(by = 'frequency', ascending = False)

    @staticmethod
    def count_words(string):
        return len(DataUtil.str_split(string))