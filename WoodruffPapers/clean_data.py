#%%
# import data
import pandas as pd
import numpy as np
import plotly.express as px
from termcolor import colored
from WoodruffData import WoodruffData
from ScriptureData import ScriptureData
from DataUtil import DataUtil
from multiprocessing import Pool


pd.set_option('display.max_rows', 100)
pd.set_option('max_colwidth', 400)
pd.options.mode.chained_assignment = None
# this powers the word_tokenize function
# nltk.download('punkt')


#%%
# read woodruff data
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
# path_woodruff = '../data/data_woodruff_raw.csv'
path_woodruff = '../data/raw_entries.csv'
data_woodruff = pd.read_csv(path_woodruff)

columns = ['date', 'text']
data_woodruff = data_woodruff[columns]
data_woodruff

# clean woodruff data
data_woodruff['text'] = data_woodruff['text'].replace(WoodruffData.symbols, regex=True)

# loop through entries and remove rows that have regex match in entry
for entry in WoodruffData.entries_to_remove:
    data_woodruff = DataUtil.regex_filter(data_woodruff, 'text', entry)

# lowercase all text
data_woodruff['text'] = data_woodruff['text'].str.lower()

# remove stop words
stop_words = list(set(stopwords.words('english')))
regex_remove = '\.|\:|\;|\,|\-|\(|\)|\?|wo'
data_woodruff['text_clean'] = data_woodruff['text'].apply(lambda x: DataUtil.str_remove(x, regex_remove))
data_woodruff['text_clean'] = data_woodruff['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data_woodruff['text_clean'] = data_woodruff['text_clean'].apply(lambda x: DataUtil.remove_duplicate_words(x))
data_woodruff
# DataUtil.create_frequency_dist(DataUtil.combine_rows(data_woodruff['text_clean'].head(5)))


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

# select only relevant columns
data_scriptures = data_scriptures[['volume_title', 'verse_title', 'scripture_text']]

data_scriptures['text_clean'] = data_scriptures['scripture_text'].apply(lambda x: DataUtil.str_remove(x, regex_remove))
data_scriptures['text_clean'] = data_scriptures['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
data_scriptures['text_clean'] = data_scriptures['text_clean'].apply(lambda x: DataUtil.remove_duplicate_words(x))
data_scriptures


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

import os
if os.path.exists("test1.csv"):
    os.remove("test1.csv")
with open("test1.csv", 'a') as f:
    f.write(string_woodruff + ',' + string_scriptures)


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
pd.DataFrame(matches)

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
for verse_text in scripture_data.data.head(2)['scripture_text']:
    print('='*50)
    verse_split = DataUtil.split_string_into_list(verse_text, 15)
    tr_idf_model  = TfidfVectorizer()
    verse_vector = tr_idf_model.fit_transform(verse_split).toarray()
    verse_tf_idf = pd.DataFrame(tf_idf_array, columns = words_set)

    # print(verse_tf_idf)
    for index, row in verse_tf_idf.iterrows():
        verse_vector = list(row)
        print(verse_vector)

#%%
v1 = [0.28680742049286617, 0.0, 0.28680742049286617, 0.21812443587607397, 0.28680742049286617, 0.0, 0.0, 0.28680742049286617, 0.0, 0.0, 0.0, 0.21812443587607397, 0.28680742049286617, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.28680742049286617, 0.28680742049286617, 0.0, 0.0, 0.0, 0.0, 0.28680742049286617, 0.0, 0.0, 0.0, 0.0, 0.28680742049286617, 0.28680742049286617, 0.28680742049286617, 0.0]
v2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.2802187187492852, 0.0, 0.0, 0.2802187187492852, 0.2802187187492852, 0.0, 0.21311355837330712, 0.0, 0.2802187187492852, 0.2802187187492852, 0.0, 0.2802187187492852, 0.0, 0.0, 0.2802187187492852, 0.0, 0.0, 0.21311355837330712, 0.2802187187492852, 0.2802187187492852, 0.0, 0.0, 0.21311355837330712, 0.2802187187492852, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2802187187492852]
cosine_similarity(v1, v2)
# woodruff_data.data_raw.to_csv(path_woodruff)

woodruff_data.clean_data()
# woodruff_data.data = woodruff_data.data.head(1)
woodruff_data.preprocess_data()
woodruff_data.data_preprocessed

#%%

text_sample = DataUtil.combine_rows(woodruff_data.data['text'].head(10))
DataUtil.create_frequency_dist(text_sample).head(100)

#%%
# read scripture data

scripture_data.clean_data()
scripture_data.preprocess_data()


# scripture_data.data.head(100)
# text_sample = DataUtil.combine_rows(scripture_data.data['scripture_text'].head(10))
# DataUtil.create_frequency_dist(text_sample).head(100)


#%%

# ## run verse matching algorithm
# data_sample = woodruff_data.data_preprocessed.head(1000)

# results = pd.DataFrame()

# for index, row in scripture_data.data.iterrows(): #.iloc[100 : 2000]
#     verse_title    = row['verse_title']
#     scripture_text = row['scripture_text']

#     data_sample['scripture_text'] = scripture_text
#     data_sample['percentage_match'] = (data_sample['phrase'].apply(AISwag.compute_match_percentage, scripture_text=scripture_text))
#     data_sample = data_sample.sort_values(by = 'percentage_match', ascending=False)

#     max_match = list(data_sample.iloc[0][['percentage_match', 'phrase']])

#     print()
#     print(verse_title, 'max match:', max_match[0])
#     print(scripture_text)
#     print('woodruff:', max_match[1])

#     # print('mean percentage', data_sample['percentage_match'].mean())

#     if data_sample['percentage_match'].max() >= .3:
#         print(colored('max match: '+ str(max_match), 'green'))

#         print(list(data_sample.columns))
#         # print(colored('attaching results', 'green'), data_sample['percentage_match'].max())
#         top_3_rows = data_sample.nlargest(3, 'percentage_match')[['phrase', 'scripture_text', 'percentage_match', 'text', 'Text Only Transcript']]
#         top_3_rows['verse_title'] = verse_title
#         # add top 3 percentage match rows to results dataframe
#         results = pd.concat([results, top_3_rows])





#%%

