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

# remove stop words
# stop_words = list(stopwords.words('english'))
# data_woodruff['text_clean'] = data_woodruff['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
# remove duplicate words
# data_woodruff['text_clean'] = data_woodruff['text_clean'].apply(lambda x: DataUtil.remove_duplicate_words(x))

#%%
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
path_scriptures = '../data/scriptures.csv'
data_scriptures = pd.read_csv(path_scriptures)

# clean scripture data
books = ["Book of Mormon", "Doctrine and Covenants", "New Testament"]#"New Testament",
books = ['Book of Mormon']

# filter down to only certain books
data_scriptures = data_scriptures[data_scriptures['volume_title'].isin(books)]
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

# number_of_documents = 2000
phrases_woodruff = phrases_woodruff[:110000]
phrases_scriptures = phrases_scriptures[:3000]


vectorizer = TfidfVectorizer()
tfidf_matrix_woodruff = vectorizer.fit_transform(phrases_woodruff)
tfidf_matrix_scriptures = vectorizer.transform(phrases_scriptures)
tfidf_matrix_woodruff



similarity_matrix = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_scriptures)
similarity_matrix

#%%
threshold = 0.5  # Adjust this threshold based on your preference
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


# len(top_phrases_woodruff)
# len(top_phrases_scriptures)
# len(similarity_scores)
data = pd.DataFrame({'phrase_woodruff':top_phrases_woodruff,
              'phrases_scriptures':top_phrases_scriptures,
              'similarity_scores' : similarity_scores}).sort_values(by='similarity_scores',ascending=False)
data.to_csv('top_matches.csv', index = False)

data
#%%

string_woodruff = DataUtil.combine_rows(data_woodruff['text_clean'])
string_scriptures = DataUtil.combine_rows(data_scriptures['text_clean'])
string_woodruff
string_scriptures

phrases_woodruff = DataUtil.split_string_into_list(string_woodruff, 10)
phrases_scriptures = DataUtil.split_string_into_list(string_scriptures, 10)

# combine entire string method
from numba import jit, cuda
# @jit(target_backend='cuda')
def percentage_match(phrase_woodruff, phrase_scripture):
    words_woodruff, words_scripture = word_tokenize(phrase_woodruff), word_tokenize(phrase_scripture)
    count = 0
    # print('length woodruff', len(words_woodruff))
    # print('length scripture', len(words_scripture))
    progress_bar = tqdm(total=len(words_woodruff), unit='item')
    for word_woodruff in words_woodruff:
        progress_bar.update(1)
        if word_woodruff in words_scripture and word_woodruff:
            count += 1
    return count / len(phrase_woodruff)

#%%
    # transformer = TfidfTransformer()
    # tfidf_matrix_woodruff = transformer.fit_transform(phrase_woodruff)
    # tfidf_matrix_verse = transformer.transform(words_scripture)
    # print(tfidf_matrix_woodruff)
    # print(tfidf_matrix_verse)
    # similarity_scores = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_verse)

    # tfidf_matrix = vectorizer.fit_transform(words_woodruff)
    # search_vector = vectorizer.transform([words_scripture])
    # similarity_scores = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_verse)
    # best_match_index = similarity_scores.argmax()
    # return similarity_scores, best_match_index
    # vectorizer.get_feature_names_out()
    # woodruff_word_match_ids   = np.unique(np.where(similarity_scores == 1)[0])
    # scriptures_word_match_ids = np.unique(np.where(similarity_scores == 1)[1])
    # percent_match_woodruff    = round(len(woodruff_word_match_ids) / len(words_woodruff), 4)
    # percent_match_scripture    = round(len(scriptures_word_match_ids) / len(words_scripture), 4)
    # return percent_match_woodruff

# percentage_match('hello my name is is', 'pizza pizza is yummy taco')
# cv = CountVectorizer()
# transformer = TfidfTransformer()
# words = word_tokenize('hello my name is')
# word_count_vector=cv.fit_transform(words)
# transformer.fit_transform(word_count_vector)

#%%
def extract_matches(string_woodruff, string_scriptures):
    words_woodruff, words_scripture = word_tokenize(string_woodruff), word_tokenize(string_scriptures)
    matches = []
    progress_bar = tqdm(total=len(words_woodruff), unit='item')

    for i in range(0, len(words_woodruff) - 2, 10):
        percentage_match = percentage_match(phrase_woodruff, phrase_scripture)
        progress_bar.update(1)
        phrase_woodruff = words_woodruff[i : i + 10]
        print(phrase_woodruff)

        if words_woodruff in words_scripture:
            string_match = words_woodruff[i-2] + ' ' + words_woodruff[i-1] + ' ' + words_woodruff[i] + ' ' + words_woodruff[i+1] + ' ' + words_woodruff[i+2]
            matches.append(string_match)
    return matches

string_woodruff = DataUtil.combine_rows(phrases_woodruff)[:300]
string_scriptures = DataUtil.combine_rows(phrases_scriptures)[:300]

# import os
filename = 'woodruff_full_strings.csv'
data = pd.DataFrame({'text_woodruff':[string_woodruff], 'text_scriptures':[string_scriptures]})
data.to_csv(filename, index=False)
data

# if os.path.exists(filename):
#     os.remove(filename)
# with open(filename, 'a') as f:
#     f.write(string_woodruff + ',' + string_scriptures)


#%%
# string_woodruff = "hello my name is porter how are you today i am pizza and you taco cake"
# string_scriptures = "pizza pizza porter"

# string_woodruff[:30]
print(string_woodruff)
print(string_scriptures)
matches = extract_matches(string_woodruff, string_scriptures)
len(matches)
matches

#%%

#%%

all_matches = pd.DataFrame()
phrases_woodruff1 = phrases_woodruff
progress_bar = tqdm(total=len(phrases_woodruff1), unit='item')
phrases_scriptures
for phrase_woodruff in phrases_woodruff1:
    for phrase_scripture in phrases_scriptures:
        # words_woodruff, words_scripture = DataUtil.str_split(phrase_woodruff), DataUtil.str_split(phrase_scripture)
        similarity = percentage_match(phrase_woodruff, phrase_scripture)
        if similarity > .1:
            print(similarity)
            print(phrase_woodruff)
            print(phrase_scripture)
            # print(colored('match: '+str(similarity), 'green'))
            top_matches = pd.DataFrame({'similarity': similarity,
                                        'phrase_woodruff':phrase_woodruff,
                                        'phrase_scripture': phrase_scripture,
                                        },index=[0])
            all_matches = pd.concat([all_matches, top_matches]).sort_values(by='similarity',ascending=False)
    # progress_bar.set_description(phrase_scripture[:5] + 'match count:')
    progress_bar.update(1)
progress_bar.close()
all_matches

#%%

top_matches = pd.DataFrame({'similarity': similarity,
                            'phrase_woodruff':phrase_woodruff,
                            'phrase_scripture': phrase_scripture,
                            },index=0)
# all_matches = pd.concat([all_matches, top_matches])
top_matches
#%%

# s1 = 'hello pizza pizza ok poop, pie'
# s2 = 'hello pizza swag'
# compute_similarity(phrases_woodruff[0], phrases_scriptures[0])
# compute_similarity(s1, s2)





#%%
data_scriptures1 = data_scriptures

progress_bar = tqdm(total=len(data_scriptures1), unit='item')
# loop through each entry and extract words contained in verses
all_matches = pd.DataFrame()
for verse, verse_title in zip(data_scriptures1['text_clean'], data_scriptures1['verse_title']):
    # progress_bar.set_description(verse_title)
    progress_bar.update(1)
    verse_words = DataUtil.str_split(verse)
    # iterate through words and count occurence of each word within entry
    for verse_word in verse_words:
        if verse_word in religious_words:
            data_woodruff['matches'] = data_woodruff['text_clean'].apply(lambda x: DataUtil.str_extract_all(x, regex = r' '+verse_word))
            data_woodruff['match_count'] = data_woodruff['text_clean'].apply(lambda x: DataUtil.str_count_occurrences(x, regex = r' '+verse_word ))
            # print(verse_word)
            if data_woodruff['match_count'].max() > 1:
                top_matches = data_woodruff.nlargest(1, 'match_count')
                top_matches['verse'] = verse_title
                # print(top_matches)
                all_matches = pd.concat([all_matches, top_matches]).sort_values(by = 'match_count', ascending=False)
                progress_bar.set_description(verse_title + 'match count:'+str(len(all_matches)))
# all_matches.to_csv('pizza.csv')
all_matches

# for entry in data_woodruff['text_clean'][:3]:
    # print(entry)
    # woodruff_data.data['scripture_word_matches'] = woodruff_data.data['text'].apply(lambda x: [word for word in x.split() if word in (verse)])
# woodruff_data.data


#%%

# create woodruff tfidf
# tr_idf_model  = TfidfVectorizer()
# tf_idf_vector = tr_idf_model.fit_transform(woodruff_words)

# # print(type(tf_idf_vector), tf_idf_vector.shape)
# tf_idf_array = tf_idf_vector.toarray()


# woodruff_tf_idf = pd.DataFrame(tf_idf_array, columns = tr_idf_model.get_feature_names_out())
# woodruff_tf_idf

# #%%

# # scripture_text = scripture_data.data['scripture_text'].head(10)
# # create scripture tfidf
# tr_idf_model  = TfidfVectorizer()
# tf_idf_vector = tr_idf_model.fit_transform(scripture_words)

# # print(type(tf_idf_vector), tf_idf_vector.shape)
# tf_idf_array = tf_idf_vector.toarray()


# scripture_tf_idf = pd.DataFrame(tf_idf_array, columns = tr_idf_model.get_feature_names_out())
# scripture_tf_idf

#%%
# def vectorize(string):
#     tr_idf_model  = TfidfVectorizer()
#     tf_idf_vector = tr_idf_model.fit_transform(scripture_words)
#     tf_idf_array = tf_idf_vector.toarray()
#     scripture_tf_idf = pd.DataFrame(tf_idf_array, columns = tr_idf_model.get_feature_names_out())
#     return scripture_tf_idf

# loop through list of 10 word entry phrases
for woodruff_phrase in woodruff_words[:2]:
    woodruff_score = 0
    # loop through list of 10 word scripture phrases
    for scripture_phrase in scripture_words:
        woodruff_words_split = DataUtil.str_split(woodruff_phrase)
        scripture_words_split = DataUtil.str_split(scripture_phrase)

        # remove stop words
        stop_words = list(set(stopwords.words('english')))
        woodruff_words_split = [word for word in woodruff_words_split if not word.lower() in stop_words]
        scripture_words_split = [word for word in scripture_words_split if not word.lower() in stop_words]

        # print('woodruff phrase:', woodruff_words_split)
        # print('scripture phrase', scripture_words_split)
        for woodruff_word in woodruff_words_split:
            if woodruff_word in scripture_words_split:
                print(woodruff_word)
                woodruff_score += 1


#%%


# similarity_score = cosine_similarity(woodruff_vector, verse_vector)
# print(similarity_score)
# print(cosine)


#%%



#%%
# testing
