#%%
import pandas as pd
from termcolor import colored
from DataUtil import DataUtil
# import cupy as cp
# from cupy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from numba import jit, float64, typeof
import os
pd.set_option('display.max_colwidth', None)

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

replacements = {
    r'(, b\.)'              : r'',
    r'\<U\+25CA\>'          : r'',
    r'\&amp;c?'             : r"and",
    r'\&apos;'              : r"'",
    r"(\^?FIGURES?\^?)"     : r'',
    r'[\{\}\~]'             : r'',
    r'\s{2}'                : r' ',
    r','                    : r'',
    # r'\[(\w+)\]'            : r'',
    r'\n'                   : r' ',
    r'\[\[(.*?)\|(.*?)\]\]' : r'\1',
    r'\-\s'                  : r'',
    # r'- ng '                 : r'ng ',
    # r' ng '                 : r'ng ',
    # r' ed '                 : r'ed ',
    r'\n'                 : r' ',
    r'\s+'                 : r' ',
    r'\.'                 : r'',
    r'\.|\:|\;|\,|\-|\(|\)|\?' : r'',
    r'sacrafice'    : r'sacrifice',
    r'discours'     : r'discourse',
    r'travling'      : r'traveling',
    r'oclock'       : r'oclock',
    r'w\. woodruff' : r'wilford woodruff',
    r'any\s?whare'    : r'anywhere',
    r'some\s?whare'     : r'somewhere',
    r'whare'         : r'where',
    r'sumthing'      : r'something',
    r' els '         : r' else ',
    r' wil '         : r' will ',
    r'savio saviour' : r'saviour',
    r'arived'       : r'arrived',
    r'intirely    ' : r'entirely',
    r'phylosophers' : r'philosophers',
    r'baptised'     : r'baptized',
    r'benef\- it'   : r'benefit',
    r'preachi \-ng'      : r'preaching',
    r'oppor- tunities' : r'opportunities',
    r'vary'         : r'very',
    r'councellor'   : r'counselor',
    r'sircumstances' : r'circumstances',
    r'Preasent'    : r'present',
    r'sept\.'      : r'september',
    r'Sacramento Sacramento' : r'Sacramento',
    r'tryed'       : r'tried',
    r'fals'        : r'false',
    r'aprail'      : r'april',
    r'untill'      : r'until',
    r'sumwhat'      : r'somewhat',
    r'joseph smith jun' : r'joseph smith jr',
    r'miricle' : r'miracle',
    r'procedings' : r'proceedings',
    r'w odruff' : r'woodruff',
    r'prefered' : r'preferred',
    r'traveling' : r'pizza',
    r'esspecially' : r'especially',
    r'ownly' : r'only',
    }

weird_facsimile_thingy = r'a – b – c d – e f – g – – h i j – – k – – l — m – – n – o p q – r s t – u – v – w – – x – y z and 1 – – 2 – 3 – 4 – 5 – – – 6 7 – – 8 – 9 – – 0 ——— – – — – – – – – – –'

#%%
os.chdir('C:/Users/porte/Desktop/coding/Wilford-Woodruff-Papers/')

# local data
path_data_woodruff_raw = 'data/raw/data_woodruff_raw.csv'
path_data_woodruff_clean = 'data/raw/data_woodruff_clean.csv'
path_data_scriptures = 'data/raw/data_scriptures.csv'
path_matches = 'data/matches/top_matches.csv'

# url data
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'

# load data
data_woodruff = pd.read_csv(path_data_woodruff_raw)
data_woodruff

data_scriptures = pd.read_csv(path_data_scriptures)
data_scriptures



#%%
# lowercase all text
data_woodruff['text'] = data_woodruff['text'].str.lower()

# clean woodruff data
data_woodruff['text'] = data_woodruff['text'].replace(replacements, regex=True)
# remove the single weird facsimile thingy its screwing everything up
data_woodruff['text'] = data_woodruff['text'].apply(lambda x: DataUtil.str_remove(x, regex = weird_facsimile_thingy))
# loop through entries and remove rows that have regex match in entry
for entry in entries_to_remove:
    data_woodruff = DataUtil.regex_filter(data_woodruff, 'text', entry)

data_woodruff.to_csv(path_data_woodruff_clean, index = False)
# DataUtil.regex_filter(data_woodruff, 'text', )

# output this just to check if cleaned data is really clean

#%%
text_woodruff = DataUtil.combine_rows(data_woodruff['text'])
phrases_woodruff = DataUtil.split_string_into_list(text_woodruff, n = 15)
print('woodruff phrase count:', len(phrases_woodruff))

phrases_woodruff

#%%

# clean scripture data
data_scriptures['scripture_text'] = data_scriptures['scripture_text'].str.lower()
data_scriptures['scripture_text'] = data_scriptures['scripture_text'].replace(replacements, regex=True)

# select only relevant columns
data_scriptures = data_scriptures[['volume_title', 'book_title', 'verse_title', 'scripture_text']]

# split each verse into a 15 word phrase then explode it all
data_scriptures['scripture_text'] = data_scriptures['scripture_text'].apply(lambda x: DataUtil.split_string_into_list(x, 15))
data_scriptures = data_scriptures.explode('scripture_text')
data_scriptures

#%%

# @jit(nopython=True, parallel=True)
def extract_matches(phrases_woodruff, tfidf_matrix_woodruff, vectorizer, phrases_scriptures):
    tfidf_matrix_scriptures = vectorizer.transform(phrases_scriptures)
    similarity_matrix = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_scriptures)
    # time.sleep(1)
    threshold = 0.65  # Adjust this threshold based on your preference
    similarity_scores = []
    top_phrases_woodruff = []
    top_phrases_scriptures = []

    for i, phrase_woodruff in enumerate(phrases_woodruff):
        for j, phrase_scriptures in enumerate(phrases_scriptures):
            similarity_score = similarity_matrix[i][j]
            # print(similarity_score)
            # print(phrase_scriptures)
            # print(phrase_woodruff)
            if similarity_score > threshold:
                top_phrases_woodruff.append(phrase_woodruff)
                top_phrases_scriptures.append(phrase_scriptures)
                similarity_scores.append(similarity_score)

    data = pd.DataFrame({
            'phrase_woodruff':top_phrases_woodruff,
            'phrase_scripture':top_phrases_scriptures,
            'similarity_score' : similarity_scores}).sort_values(by='similarity_score',ascending=False)
    return data


#%%

# iterate through books and run model.
volume_titles = [
     'Old Testament',
     'New Testament',
     'Book of Mormon',
     'Doctrine and Covenants',
     'Pearl of Great Price',
     ]
data_scriptures1 = data_scriptures.query("volume_title in @volume_titles")
data_scriptures1

#%%
vectorizer = TfidfVectorizer()
tfidf_matrix_woodruff = vectorizer.fit_transform(phrases_woodruff)

progress_bar = tqdm(total=len(data_scriptures1))
total_matches = pd.DataFrame()
# loop through each 15 word phrase of scriptures dataset and run algorithm against it.
for i in range(len(data_scriptures1)):
    row = list(data_scriptures1.iloc[i])
    volume_title = row[0]
    book_title = row[1]
    verse_title = row[2]
    phrase_scripture = [row[3]]

    # tfidf_matrix_type = typeof(tfidf_matrix_woodruff)
    # vectorizer_type = typeof(vectorizer)
    # extract_matches(phrases_woodruff, tfidf_matrix_woodruff.astype(tfidf_matrix_type), vectorizer.astype(vectorizer_type), phrase_scripture)

    # extract scripture phrase matches from woodruff journal entry list of phrases
    top_matches = extract_matches(phrases_woodruff, tfidf_matrix_woodruff, vectorizer, phrase_scripture)
    top_matches['verse_title'] = verse_title
    top_matches['volume_title'] = volume_title
    total_matches = pd.concat([total_matches, top_matches]).sort_values(by = 'similarity_score',ascending=False)
    total_matches.to_csv(path_matches, index = False)

    progress_bar.update(1)
    description = verse_title + ' total match count: ' + str(len(total_matches))# + 'verse length: ' + str(len(phrases_scriptures[0]))
    progress_bar.set_description(description)

progress_bar.close()

total_matches



#%%
