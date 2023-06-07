#%%
import re
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from termcolor import colored
from DataUtility import DataUtility


pd.set_option('display.max_rows', 100)
pd.options.display.max_colwidth = 200
pd.options.mode.chained_assignment = None
# this powers the word_tokenize function
nltk.download('punkt')



#%%
# read in data
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"


data_scriptures = pd.read_csv(url_scriptures)
data_woodruff = pd.read_csv(url_woodruff)

data_woodruff = data_woodruff.query("`Document Type` == 'Journals'")

# data_woodruff2 = pd.read_csv('../data/data_woodruff.csv')
data_scriptures
data_woodruff

#%%


class Scriptures:

    def __init__(self, data_scriptures) -> None:
        self.data_scriptures = data_scriptures

    def clean(self):
        books = ["New Testament","Book of Mormon", "Docrtine and Covenants", "Pearl of Great Price"]
        self.data_scriptures = self.data_scriptures.query('volume_title == @books')
        self.data_scriptures['scripture_text'] = self.data_scriptures['scripture_text'].str.lower()


class WoodruffPapers:

    def __init__(self, data_woodruff, sample = True) -> None:
        if sample:
            self.data_woodruff = data_woodruff.sample(100)
        else:
            self.data_woodruff = data_woodruff

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
        string = DataUtility.replace_regex(string, r'\&apos;', r"'")
        string = DataUtility.replace_regex(string, r'\&amp;c', r"'")
        string = DataUtility.replace_regex(string, r'\&amp;', r'and')
        # print(len(re.findall(r'\&amp;', string)))
        return string

    @staticmethod
    def remove_birthyears(string):
        string = re.sub(r'(, b\.)', string)
        return string

    @staticmethod
    def fix_typos(string):
        DataUtility.replace_regex('travling', regex = r'', replacement = r'')

    def clean(self):
        # rename Text Only Transcript to 'text'
        self.data_woodruff = self.data_woodruff.rename(columns={"Text Only Transcript": "text"})
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(DataUtility.replace_regex, regex = r'\[(\w+)\]')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(DataUtility.replace_regex, regex = r',', replacement = r'')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(DataUtility.replace_regex, regex = r'[\{\}\~]', replacement = r'')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(DataUtility.replace_regex, regex = r'\s{2}', replacement = r' ')
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(DataUtility.remove_new_lines)
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.clean_suggestions)
        self.data_woodruff['text'] = self.data_woodruff['text'].apply(WoodruffPapers.clean_ampersands)
        # TODO: implement typos and symbols
        typos = {'sacrafice':'sacrifice',
                 'discours':'discourse'}
        symbols = {
                'b. 1795':'',
                '<U+25CA>':'',
                '&amp;':'and',
                '&amp;c':'and',
                '&apos;':"'",
                '[FIGURE]':'',
                'discours':'discourse'
            }
        things_to_remove = ['WW 1841-2',
                            'Front cover',
                            'THE SECOND BOOK OF WILLFORD FOR 1839',
                            'W\. WOODRUFFs DAILY JOURNAL AND HISTORY IN 1842',
                            "WILFORD WOODRUFF's DAILY JOURNAL AND HISTORY IN 1843",
                            "WILLFORD WOODRUFF'S JOURNAL VOL\. 2\. AND A SYNOPSIS OF VOL\. 1\.",
                            "Willford Woodruff's Journal Containing an Account Of my life and travels from the time of my first connextion with the Church of Jesus Christ of Latter-day Saints",
                            'THE FIRST BOOK OF WILLFORD VOL\. 2\. FOR 1838',
                            r"WILLFORD\. WOORUFF's DAILY JOURNAL AND TRAVELS",
                            r"(\^?FIGURES?\^?)"]

        for regex in things_to_remove:
            self.data_woodruff = DataUtility.regex_filter(self.data_woodruff, column = 'text', regex=regex)
        # lowercase all entries
        self.data_woodruff['text'] = self.data_woodruff['text'].str.lower()

    def preprocess(self):
        self.data_woodruff['phrase'] = self.data_woodruff['text'].apply(DataUtility.split_string_into_list, n = 15)
        # explode dataset so each row contains a single 15 word phrase
        self.data_woodruff = self.data_woodruff.explode('phrase')

    @staticmethod
    def compute_match_percentage(text_woodruff, text_scripture):
        words_woodruff = DataUtility.split_string(text_woodruff)
        words_scripture = DataUtility.split_string(text_scripture)
        vectorizer = TfidfVectorizer()
        ### vectorize words
        # Compute TF-IDF matrices
        tfidf_matrix_woodruff = vectorizer.fit_transform(words_woodruff)
        tfidf_matrix_verse = vectorizer.transform(words_scripture)

        similarity_scores = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_verse)
        vectorizer.get_feature_names_out()
        woodruff_word_match_ids   = np.unique(np.where(similarity_scores == 1)[0])
        scriptures_word_match_ids = np.unique(np.where(similarity_scores == 1)[1])
        percent_match_woodruff    = round(len(woodruff_word_match_ids) * 100 / len(words_woodruff), 2)
        return percent_match_woodruff

woodruff_papers = WoodruffPapers(data_woodruff, sample = True)
woodruff_papers.clean()

scriptures = Scriptures(data_scriptures)
scriptures.clean()

scriptures.data_scriptures
woodruff_papers.data_woodruff


#%%

woodruff_papers.preprocess()

woodruff_papers.data_woodruff

#%%
from tqdm import tqdm
results = pd.DataFrame()

for i in tqdm(range(10), desc='processing'):
    verse_title = list(woodruff_papers.data_scriptures['verse_title'])[i]
    text_scripture = list(woodruff_papers.data_scriptures.query('verse_title == @verse_title')['scripture_text'])[0]

    # print('comparing:', verse_title)

    data_sample['text_scripture'] = text_scripture
    data_sample['percentage_match'] = (data_sample['phrase'].apply(WoodruffPapers.compute_match_percentage,
                                                        text_scripture=text_scripture))
    data_sample.sort_values(by = 'percentage_match', ascending=False)

    if data_sample['percentage_match'].max() > 50:
        # print(colored('attaching results', 'green'), data_sample['percentage_match'].max())
        top_3_rows = data_sample.nlargest(3, 'percentage_match')[['phrase', 'text_scripture', 'percentage_match']]
        results = pd.concat([results, top_3_rows])


#%%
data_sample.to_csv('sample.csv', index = False)



#%%
## EDA
text_woodruff = DataUtility.combine_rows(woodruff_papers.data_woodruff.head(1000)['text'])
text_verse = DataUtility.combine_rows(woodruff_papers.data_scriptures.head(1000)['scripture_text'])

print(text_verse)
print(text_woodruff)
freq = DataUtility.create_frequency_distribution(text_woodruff)
print(freq.head(100))
freq.tail(100)

chart = px.bar(freq.head(100), x='frequency', y='word', text_auto='.2s',orientation='h')
chart.show()

# chart = px.bar(freq.tail(100), x='frequency', y='word', text_auto='.2s',orientation='h')
# chart.show()

# chart = px.bar(freq.tail(100), x='frequency', y='word', text_auto='.2s',orientation='h')
# chart.show()


#%%