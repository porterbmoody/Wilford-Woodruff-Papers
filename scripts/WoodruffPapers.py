#%%
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from termcolor import colored
pd.set_option('display.max_rows', 100)
pd.options.mode.chained_assignment = None
# this powers the word_tokenize function
nltk.download('punkt')


#%%
# read in data
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
raw_dat = pd.read_csv("https://github.com/wilfordwoodruff/Public_Stories/raw/main/data/derived/derived_data.csv")
data_scriptures = pd.read_csv(url_scriptures)
data_woodruff = raw_dat.query("`Document Type` == 'Journals'")

data_woodruff2 = pd.read_csv('../data/data_woodruff.csv')
data_scriptures
data_woodruff


#%%
## class with static utilities for cleaning the entries
class WoodruffPapers:


    def __init__(self, data_woodruff, data_scriptures) -> None:
        self.data_woodruff = data_woodruff
        self.data_scriptures = data_scriptures

    def clean_scriptures(self):
        self.data_scriptures = self.data_scriptures.query('volume_title in ["New Testament","Book of Mormon", "Docrtine and Covenants"]')

    # clean suggestions
    @staticmethod
    def clean_suggestions(string):
        pattern = r'\[\[(.*?)\]\]'
        suggestions = re.findall(pattern, string)
        for suggestion in suggestions:
            replacement = suggestion.split('|')[0] # split the words before and after the '|'
            suggestion_exact_match = '[[' + suggestion + ']]'
            string = string.replace(suggestion_exact_match, replacement)
        return string

    @staticmethod
    def clean_ampersands(string):
        string = WoodruffPapers.replace_regex(string, r'\&apos;', r"'")
        string = WoodruffPapers.replace_regex(string, r'\&amp;c', r"'")
        string = WoodruffPapers.replace_regex(string, r'\&amp;', r'and')
        # print(len(re.findall(r'\&amp;', string)))
        return string

    @staticmethod
    def remove_birthyears(string):
        string = re.sub(r'(, b\.)', string)
        return string

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
    def fix_typos(self):
        self.data_woodruff = re.sub(self.data_woodruff)
        print(self.data_woodruff)

    @staticmethod
    def create_frequency_distribution(string):
        string = string.lower()
        tokens = word_tokenize(string)
        frequency_distribution = FreqDist(tokens)
        data = pd.DataFrame(list(frequency_distribution.items()), columns=['word', 'frequency'])
        print(data.shape)
        return data.sort_values(by = 'frequency', ascending = False)

    @staticmethod
    def count_word_frequency(search_word):
        print(search_word)

    @staticmethod
    def remove_new_lines(string):
        return re.sub(r'\n', r' ', string)

    @staticmethod
    def replace_regex(string, regex, replacement = None):
        if replacement is None:
            string = re.sub(regex, r'\1', string)
            return string
        string = re.sub(regex, replacement, string)
        return string

    @staticmethod
    def fix_typos(string):
        WoodruffPapers.replace_regex('travling', regex = r'', replacement = r'')

    @staticmethod
    def regex_filter(dataframe, column, regex):
        return dataframe[dataframe[column].str.contains(regex) == False]

    def clean(self):
        # rename Text Only Transcript to 'text'
        self.data_woodruff = self.data_woodruff.rename(columns={"Text Only Transcript": "text"})
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.replace_regex, regex = r'\[(\w+)\]')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.replace_regex, regex = r',', replacement = r'')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.replace_regex, regex = r'[\{\}\~]', replacement = r'')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.replace_regex, regex = r'\s{2}', replacement = r' ')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.remove_new_lines)
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.clean_suggestions)
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.clean_ampersands)

        typos = {'sacrafice':'sacrifice'}
        things_to_remove = ['WW 1841-2',
                            'Front cover',
                            'THE SECOND BOOK OF WILLFORD FOR 1839',
                            'W\. WOODRUFFs DAILY JOURNAL AND HISTORY IN 1842',
                            "WILFORD WOODRUFF's DAILY JOURNAL AND HISTORY IN 1843",
                            "WILLFORD WOODRUFF'S JOURNAL VOL\. 2\. AND A SYNOPSIS OF VOL\. 1\.",
                            "Willford Woodruff's Journal Containing an Account Of my life and travels from the time of my first connextion with the Church of Jesus Christ of Latter-day Saints",
                            'THE FIRST BOOK OF WILLFORD VOL\. 2\. FOR 1838',
                            r"WILLFORD\. WOORUFF's DAILY JOURNAL AND TRAVELS",
                            r"\^?FIGURES\^?"]

        for regex in things_to_remove:
            self.data_woodruff = self.regex_filter(self.data_woodruff, column = 'text', regex=regex)

    @staticmethod
    def compute_match_percentage(text_woodruff, text_scriptures):
        words_woodruff = WoodruffPapers.split_string(text_woodruff)
        words_scriptures = WoodruffPapers.split_string(text_scriptures)
        vectorizer = TfidfVectorizer()
        ### vectorize words
        # Compute TF-IDF matrices
        tfidf_matrix_woodruff = vectorizer.fit_transform(words_woodruff)
        tfidf_matrix_verse = vectorizer.transform(words_scriptures)

        similarity_scores = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_verse)
        vectorizer.get_feature_names_out()
        woodruff_word_match_ids   = np.unique(np.where(similarity_scores == 1)[0])
        scriptures_word_match_ids = np.unique(np.where(similarity_scores == 1)[1])
        percent_match_woodruff    = round(len(woodruff_word_match_ids) * 100 / len(words_woodruff), 2)
        return percent_match_woodruff


woodruff_papers = WoodruffPapers(data_woodruff, data_scriptures)
woodruff_papers.clean()
woodruff_papers.clean_scriptures()
woodruff_papers.data_woodruff

#%%
# x = "WILLFORD WOODRUFF'S JOURNAL VOL. 2. AND A SYNOPSIS OF VOL. 1."
# woodruff_papers.data_woodruff.query("text == @x")

#%%
text_woodruff = WoodruffPapers.combine_rows(woodruff_papers.data_woodruff.head(1000)['text'])
text_verse = WoodruffPapers.combine_rows(woodruff_papers.data_scriptures.head(1000)['scripture_text'])

print(text_verse)
print(text_woodruff)
freq = WoodruffPapers.create_frequency_distribution(text_woodruff)
print(freq.head(100))
freq.tail(100)
#%%
# WoodruffPapers.count_word_frequency(search_word='god')
# woodruff_papers.data_woodruff


#%%
## EDA
chart = px.bar(freq.head(100), x='frequency', y='word', text_auto='.2s',orientation='h')
chart.show()

chart = px.bar(freq.tail(100), x='frequency', y='word', text_auto='.2s',orientation='h')
chart.show()

chart = px.bar(freq.tail(100), x='frequency', y='word', text_auto='.2s',orientation='h')
chart.show()


#%%
# for i in range(0, 50, 10):
#     for j in range(0, 100, 10):
#         current_words_woodruff = words_woodruff[i : i + 10]
#         current_words_scriptures = words_verse[j : j + 20]
#         compute_match_percentage(current_words_woodruff, current_words_scriptures)

# extract verse
data_sample = woodruff_papers.data_woodruff.sample(1000)
verses = []

def split_string_into_list(text, n):
    words = text.split()
    result = [' '.join(words[i:i+n]) for i in range(0, len(words), n)]
    return result



#%%

for i in range(30):
    verse = list(woodruff_papers.data_scriptures['verse_title'])[i]
    text_scriptures = list(woodruff_papers.data_scriptures.query('verse_title == @verse')['scripture_text'])[0]
    text_scriptures
    print('comparing:', verse)


    data_sample[verse] = (data_sample['text'].apply(WoodruffPapers.compute_match_percentage,
                                                    text_scriptures=text_scriptures))
    verses.append(verse)
    print(data_sample[verse].max())
    # print(data_sample[data_sample[verse] > 20])

    if data_sample[verse].max() > 20:
        print()
        print(colored('top matches:', 'green'))
        # print('verse:', text_scriptures)
        print(data_sample.sort_values(by = verse, ascending=False)[[verse, 'text']].head(6))

print(verses)

#%%

#%%
data_sample.to_csv('sample.csv', index = False)


