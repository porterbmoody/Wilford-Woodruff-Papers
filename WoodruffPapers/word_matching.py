#%%
import pandas as pd
import numpy as np
import plotly.express as px
from termcolor import colored
from WoodruffData import WoodruffData
from ScriptureData import ScriptureData
from DataUtil import DataUtil
# from AISwag import AISwag
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
pd.set_option('display.max_colwidth', None)

nltk.download('stopwords')

entries_to_remove = [
    r'WW 1841-2',
    r'Front cover',
    r'THE SECOND BOOK OF WILLFORD FOR 1839',
    r'W\. WOODRUFFs DAILY JOURNAL AND HISTORY IN 1842',
    r"WILFORD WOODRUFF's DAILY JOURNAL AND HISTORY IN 1843",
    r"WILLFORD WOODRUFF'S JOURNAL VOL\. 2\. AND A SYNOPSIS OF VOL\. 1\.",
    r"Willford Woodruff's Journal Containing an Account Of my life and travels from the time of my first connextion with the Church of Jesus Christ of Latter-day Saints",
    r'THE FIRST BOOK OF WILLFORD VOL\. 2\. FOR 1838',
    r"WILLFORD\. WOORUFF's DAILY JOURNAL AND TRAVELS",
    r'Pgs 172\–288 are blank',
    ]

symbols = {
        # 'b\. 1795':'',
        r'(, b\.)'              : r'',
        r'\<U\+25CA\>'          : r'',
        r'\&amp;c?'             : r"and",
        r'\&apos;'              : r"'",
        r"(\^?FIGURES?\^?)"     : r'',
        r'[\{\}\~]'             : r'',
        r'\s{2}'                : r' ',
        r','                    : r'',
        r'\[(\w+)\]'            : r'',
        r'\n'                   : r' ',
        r'\[\[(.*?)\|(.*?)\]\]' : r'\1',
        r'\-\s'                  : r'',
        r'- ng '                 : r'ng ',
        r' ng '                 : r'ng ',
        r' ed '                 : r'ed ',
        r'\n'                 : r' ',
        r'\s+'                 : r' ',
        r'\.'                 : r'',
        r'\.|\:|\;|\,|\-|\(|\)|\?|wo'                 : r'',
    }

typos = {
        r'sacrafice'    : r'sacrifice',
        r'discours'     : r'discourse',
        r'travling'      : r'traveling',
        r'oclock'       : r'oclock',
        r'[Ww]\. [Ww]oodruff' : r'Wilford Woodruff',
        r'any\s?whare'    : r'anywhere',
        r'some\s?whare'     : r'somewhere',
        r'whare'         : r'where',
        r'sumthing'      : r'something',
        r' els '         : r' else ',
        r'savio saviour' : r'saviour',
        r'arived' : r'arrived',
        r'intirely    ' : r'entirely',
        r'phylosophers' : r'philosophers',
        r'baptised'     : r'baptized',
        r'benef\- it'   : r'benefit',
        r'preachi \-ng'      : r'preaching',
        r'oppor- tunities' : r'opportunities',
        r'vary'         : r'very',
        r'Councellor'   : r'Counselor',
        r'councellor'   : r'counselor',
        r'sircumstances' : r'circumstances',
        r'Preasent'    : r'present',
        r'Sept\.'      : r'September',
        r'Sacramento Sacramento' : r'Sacramento',
        r'tryed'       : r'tried',
        r'fals'        : r'false',
        r'Aprail'      : r'April',
        r'untill'      : r'until',
        r'sumwhat'      : r'somewhat',
        r'joseph smith jun' : r'joseph smith jr',
        r'miricle' : r'miracle',
        r'procedings' : r'proceedings',
        r'w odruff' : r'woodruff',
        r'prefered' : r'preferred',
        r'traveling' : r'pizza',
        r'esspecially' : r'especially',
        }
#%%
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
path_woodruff = '../data/raw_entries.csv'
data_woodruff = pd.read_csv(path_woodruff)

data_woodruff


columns = ['date', 'text']
data_woodruff = data_woodruff[columns]


# lowercase all text
data_woodruff['text'] = data_woodruff['text'].str.lower()

# clean woodruff data
data_woodruff['text'] = data_woodruff['text'].replace(symbols, regex=True)
data_woodruff['text'] = data_woodruff['text'].replace(typos, regex=True)

# loop through entries and remove rows that have regex match in entry
for entry in entries_to_remove:
    data_woodruff = DataUtil.regex_filter(data_woodruff, 'text', entry)


data_woodruff.to_csv('test.csv', index = False)

text_woodruff = DataUtil.combine_rows(data_woodruff['text'])
phrases_woodruff = DataUtil.split_string_into_list(text_woodruff, n = 15)

print('scripture woodruff:', len(phrases_woodruff))
phrases_woodruff


#%%
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
path_scriptures = '../data/scriptures.csv'
data_scriptures = pd.read_csv(path_scriptures)

# clean scripture data
books = ["Book of Mormon", "Doctrine and Covenants", "New Testament"]#"New Testament",
book = books[2]

# filter down to only certain books
data_scriptures = data_scriptures[data_scriptures['volume_title'].isin([book])]
data_scriptures['scripture_text'] = data_scriptures['scripture_text'].str.lower()
data_scriptures['scripture_text'] = data_scriptures['scripture_text'].replace(typos, regex=True)

# select only relevant columns
data_scriptures = data_scriptures[['volume_title', 'verse_title', 'scripture_text']]

# data_scriptures['text_clean'] = data_scriptures['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
# data_scriptures['text_clean'] = data_scriptures['text_clean'].apply(lambda x: DataUtil.remove_duplicate_words(x))
data_scriptures

text_scriptures = DataUtil.combine_rows(data_scriptures['scripture_text'])
phrases_scriptures = DataUtil.split_string_into_list(text_scriptures, n = 15)
print('scripture phrases:', len(phrases_scriptures))
phrases_scriptures


#%%




def extract_matches(phrases_woodruff, phrases_scriptures):
    vectorizer = TfidfVectorizer()
    tfidf_matrix_woodruff = vectorizer.fit_transform(phrases_woodruff)
    tfidf_matrix_scriptures = vectorizer.transform(phrases_scriptures)
    tfidf_matrix_woodruff

    similarity_matrix = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_scriptures)
    similarity_matrix


    threshold = 0.6  # Adjust this threshold based on your preference
    similarity_scores = []
    top_phrases_woodruff = []
    top_phrases_scriptures = []
    progress_bar = tqdm(total=len(phrases_woodruff), unit='item')

    for i, phrase_woodruff in enumerate(phrases_woodruff):
        progress_bar.update(1)
        for j, phrase_scriptures in enumerate(phrases_scriptures):
            similarity_score = similarity_matrix[i][j]
            if similarity_score > threshold:
                top_phrases_woodruff.append(phrase_woodruff)
                top_phrases_scriptures.append(phrase_scriptures)
                similarity_scores.append(similarity_score)

    progress_bar.close()
    data = pd.DataFrame({'phrase_woodruff':top_phrases_woodruff,
                'phrases_scriptures':top_phrases_scriptures,
                'similarity_scores' : similarity_scores}).sort_values(by='similarity_scores',ascending=False)
    return data


# number_of_documents = 2000
n = 8000
# for n in len(phrases_scriptures)
# phrases_woodruff = phrases_woodruff
# phrases_scriptures = phrases_scriptures[min : max]
# phrases_scriptures

for n in range(0, len(phrases_scriptures), 4000):
    min, max = n, n + 4000
    if max > len(phrases_scriptures):
        min, max = n, len(phrases_scriptures)
    print('batch', min, max)
    path_matches = '../matches/top_matches_' + book + '_' + str(min) + '_to_' + str(max) + '_verses.csv'
    # path_matches
    data = extract_matches(phrases_woodruff, phrases_scriptures[min:max])
    data.to_csv(path_matches, index = False)

    print(data)








#%%

from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer

# normal function to run on cpu
def func(a):
    for i in range(100000000):
        a[i]+= 1

# function optimized to run on gpu
@jit(target_backend='cuda')
def func2(a):
    for i in range(100000000):
        a[i]+= 1
if __name__=="__main__":
    n = 100000000
    a = np.ones(n, dtype = np.float64)

    start = timer()




    func(a)
    print("without GPU:", timer()-start)

    start = timer()
    func2(a)
    print("with GPU:", timer()-start)


# %%
