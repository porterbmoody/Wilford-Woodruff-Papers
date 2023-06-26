#%%
# import data
import pandas as pd
import numpy as np
import plotly.express as px
from termcolor import colored
from WoodruffData import WoodruffData
from ScriptureData import ScriptureData
from DataUtil import DataUtil
from AISwag import AISwag
from multiprocessing import Pool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


pd.set_option('display.max_rows', 100)
pd.set_option('max_colwidth', 400)
pd.options.mode.chained_assignment = None
# this powers the word_tokenize function
# nltk.download('punkt')


#%%
# read woodruff data
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
path_woodruff = '../data/data_woodruff_raw.csv'
woodruff_data = WoodruffData(path_woodruff)


# woodruff_data.data_raw.to_csv(path_woodruff)

woodruff_data.clean_data()
woodruff_data.data


# woodruff_data.data = woodruff_data.data.head(1)
# woodruff_data.preprocess_data()


#%%
# read scripture data
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
path_scriptures = '../data/scriptures.csv'
scripture_data = ScriptureData(path_scriptures)

scripture_data.clean_data()

scripture_data.data
#%%
text_sample = DataUtil.combine_rows(woodruff_data.data['text'].head(10))
DataUtil.create_frequency_dist(text_sample).head(100)

# woodruff_strings = DataUtil.split_string_into_list(text_sample, n = 15)
# DataUtil.create_frequency_dist(text_sample).head(100)

woodruf_text = DataUtil.combine_rows(woodruff_data.data['text'].head(10))
woodruff_strings = DataUtil.split_string_into_list(woodruf_text, n = 15)
woodruff_strings


scripture_text = DataUtil.combine_rows(woodruff_data.data['text'].head(20))
scripture_strings = DataUtil.split_string_into_list(scripture_text, n = 15)
scripture_strings


#%%

vectorizer = TfidfVectorizer(stop_words='english')


# words_woodruff = DataUtil.str_split(text_woodruff)
# words_scripture = DataUtil.str_split(scripture_text)

# Compute TF-IDF matrices
tfidf_matrix_woodruff = vectorizer.fit_transform(woodruff_strings)
# tfidf_matrix_verse = vectorizer.transform(woodruff_strings)
df_idf = pd.DataFrame(vectorizer, index=vectorizer.get_feature_names(),columns=["idf_weights"])
df_idf
# similarity_scores = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_verse)
# similarity_scores

# vectorizer.get_feature_names_out()
# woodruff_word_match_ids   = np.unique(np.where(similarity_scores == 1)[0])
# scriptures_word_match_ids = np.unique(np.where(similarity_scores == 1)[1])
# percent_match_woodruff    = round(len(woodruff_word_match_ids) / len(words_woodruff), 4)
# percent_match_scripture    = round(len(scriptures_word_match_ids) / len(words_scripture), 4)
# return percent_match_woodruff

#%%



#%%

# scripture_data.preprocess_data()



scripture_data.data.head(100)
text_sample = DataUtil.combine_rows(scripture_data.data['scripture_text'].head(10))
DataUtil.create_frequency_dist(text_sample).head(100)


#%%

## run verse matching algorithm
data_sample = woodruff_data.data_preprocessed.head(1000)

results = pd.DataFrame()

for index, row in scripture_data.data.iterrows(): #.iloc[100 : 2000]
    verse_title    = row['verse_title']
    scripture_text = row['scripture_text']

    data_sample['scripture_text'] = scripture_text
    data_sample['percentage_match'] = (data_sample['phrase'].apply(AISwag.compute_match_percentage, scripture_text=scripture_text))
    data_sample = data_sample.sort_values(by = 'percentage_match', ascending=False)

    max_match = list(data_sample.iloc[0][['percentage_match', 'phrase']])

    print()
    print(verse_title, 'max match:', max_match[0])
    print(scripture_text)
    print('woodruff:', max_match[1])

    # print('mean percentage', data_sample['percentage_match'].mean())

    if data_sample['percentage_match'].max() >= .3:
        print(colored('max match: '+ str(max_match), 'green'))

        print(list(data_sample.columns))
        # print(colored('attaching results', 'green'), data_sample['percentage_match'].max())
        top_3_rows = data_sample.nlargest(3, 'percentage_match')[['phrase', 'scripture_text', 'percentage_match', 'text', 'Text Only Transcript']]
        top_3_rows['verse_title'] = verse_title
        # add top 3 percentage match rows to results dataframe
        results = pd.concat([results, top_3_rows])




#%%
results.sort_values(by = 'percentage_match', ascending=False)

# results


# %%

# results.sort_values(by = 'percentage_match', ascending=False)
from AISwag import AISwag
AISwag.compute_match_percentage('yo my name is porter commandment eternity',
                                'yo whats up commandment eternity')




# %%

import time

# Create a list of texts to update
texts = ["Loading", "Updating", "Processing", "Saving"]

# Create a progress bar using tqdm
from tqdm import tqdm
progress_bar = tqdm(total=len(texts), desc="Progress", unit="step")

# Iterate through the texts
for text in texts:
    # Update the progress bar description
    progress_bar.set_description(text)

    # Simulate some work being done
    time.sleep(1)

    # Increment the progress bar
    progress_bar.update(1)

# Close the progress bar
progress_bar.close()


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

######### broooooooo
#%%


    # print(list(row))
    # print('verse_title:', verse_title)
    # text_scriptures = DataUtil.combine_rows(data_scriptures1[data_scriptures1['verse_title'].isin([verse_titles])]['scripture_text'])
    # splitting verse into 15 word phrases
    # print('woodruff phrase count:', len(phrases_scriptures))
    # for n in range(0, len(phrases_scriptures), increment):
        # min, max = n, n + increment
        # if max > len(phrases_scriptures):
            # min, max = n, len(phrases_scriptures)
        # print('batch', min, max)
        # path_matches

        # print('total matches found:', len(total_matches))

# progress_bar = tqdm(total=len(data_scriptures1), unit='item')

# # for index, row in data_scriptures1.iterrows():
# for i in tqdm(range(len(data_scriptures1))):
#     # get ith row of scripture data
#     row = list(data_scriptures1.iloc[i])
#     verse_title = row[1]
#     single_verse = row[2]
#     vector_length = len(single_verse.split())



#     description = verse_title + ' scripture phrase length: ' + str(len(phrases_scriptures))
#     progress_bar.set_description(description)
#     progress_bar.update(1)

#     top_matches = extract_matches(phrases_woodruff, phrases_scriptures)
#     top_matches['verse_title'] = verse_title

# progress_bar.close()

# top_matches



# from numba import jit, cuda
# import numpy as np
# # to measure exec time
# from timeit import default_timer as timer

# # normal function to run on cpu
# def func(a):
#     for i in range(100000000):
#         a[i]+= 1

# # function optimized to run on gpu
# @jit(target_backend='cuda')
# def func2(a):
#     for i in range(100000000):
#         a[i]+= 1
# if __name__=="__main__":
#     n = 100000000
#     a = np.ones(n, dtype = np.float64)

#     start = timer()
#     func(a)
#     print("without GPU:", timer()-start)

#     start = timer()
#     func2(a)
#     print("with GPU:", timer()-start)


# %%
