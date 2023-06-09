#%%
# import data
import pandas as pd
import numpy as np
import re
import plotly.express as px
from termcolor import colored
from WoodruffData import WoodruffData
from ScriptureData import ScriptureData
from DataUtil import DataUtil


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

woodruff_data.data_woodruff_raw
# woodruff_data.data_woodruff_raw.to_csv(path_woodruff)


#%%
woodruff_data.clean_data()
woodruff_data.data_woodruff



#%%
# read scripture data
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
path_scriptures = '../data/scriptures.csv'
scripture_data = ScriptureData(path_scriptures)

scripture_data.clean_data()


scripture_data.data_scriptures.head(50)



#%%
DataUtil.create_frequency_dist(string = woodruff_data.data_woodruff['text'][0]).tail(100)



#%%

## data preprocessing

# """ Preprocess the pandas dataframe for model training.
# This entails selecting only relevant columns,
# then expanding the text column into phrases of size n (in this case 15)
# First is separates each entry into a list where each element contains a 15 word string.
# Then it explodes the dataframe so that each element of the list is mapped to its own row
# So that each row contains a single 15 word phrase of an entry
# """
data_woodruff_preprocessed = data_woodruff
data_scriptures_preprocessed = data_scriptures_raw[['volume_title', 'verse_title', 'scripture_text']]

stop_words = [' and ', r' the', ' that ',
              ' of ', ' to ',
              ' with ', ' at ', ' by ', ' in ',
              ' on ',
              ' for ',
              ' us ',
              ' we ',
              ' my ',
              ' his ',
              r' \. ',
              r' i '
              r' \, ']


# remove stopwords


data_woodruff_preprocessed['text'] = data_woodruff_preprocessed['text'].apply(DataUtil.str_replace_list, regex_list = stop_words, replacement = ' ')

# text_sample = DataUtil.str_replace_list(text_sample,
#                                        regex_list = common_words,
#                                        replacement=' ')

data_woodruff_preprocessed['text'] = data_woodruff_preprocessed['text'].str.lower()

data_woodruff_preprocessed['phrase'] = data_woodruff_preprocessed['text'].apply(DataUtil.split_string_into_list, n = 15)
data_woodruff_preprocessed = data_woodruff_preprocessed.explode('phrase')

# data_woodruff.head(100)

data_scriptures_preprocessed['scripture_text'] = data_scriptures_preprocessed['scripture_text'].apply(DataUtil.str_replace_list, regex_list = stop_words, replacement = ' ')
data_scriptures_preprocessed


# count number of words in 'text'
data_woodruff_preprocessed['word_count'] = data_woodruff_preprocessed['phrase'].apply(DataUtil.count_words)

# we'll just remove all these cuz they're weird
data_woodruff_preprocessed.query('word_count < 5').head(100)

data_woodruff_preprocessed = data_woodruff_preprocessed.query('word_count > 5')
data_woodruff_preprocessed



#%%

## run verse matching algorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data_woodruff_sample = data_woodruff_preprocessed.sample(1000)

class MLModel:

    @staticmethod
    def compute_match_percentage(text_woodruff, scripture_text):
        words_woodruff = DataUtil.str_split(text_woodruff)
        words_scripture = DataUtil.str_split(scripture_text)
        vectorizer = TfidfVectorizer()
        ### vectorize words
        # Compute TF-IDF matrices
        tfidf_matrix_woodruff = vectorizer.fit_transform(words_woodruff)
        tfidf_matrix_verse = vectorizer.transform(words_scripture)

        similarity_scores = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_verse)
        vectorizer.get_feature_names_out()
        woodruff_word_match_ids   = np.unique(np.where(similarity_scores == 1)[0])
        scriptures_word_match_ids = np.unique(np.where(similarity_scores == 1)[1])
        percent_match_woodruff    = round(len(woodruff_word_match_ids) / len(words_woodruff), 4)
        return percent_match_woodruff


results = pd.DataFrame()

for index, row in data_scriptures_preprocessed.iloc[300 : 310].iterrows():
    verse_title    = row['verse_title']
    scripture_text = row['scripture_text']
    print()
    print(verse_title)
    print(scripture_text)

    data_woodruff_sample['scripture_text'] = scripture_text
    data_woodruff_sample['percentage_match'] = (data_woodruff_sample['phrase'].apply(MLModel.compute_match_percentage, scripture_text=scripture_text))
    data_woodruff_sample.sort_values(by = 'percentage_match', ascending=False)

    print('max match:', list(data_woodruff_sample.sort_values(by = 'percentage_match', ascending=True).iloc[0][['phrase', 'percentage_match']]))
    print('mean percentage', data_woodruff_sample['percentage_match'].mean())
    # print(data_woodruff_sample['percentage_match'].max())

    if data_woodruff_sample['percentage_match'].max() > 50:
        print(colored('attaching results', 'green'), data_woodruff_sample['percentage_match'].max())
        top_3_rows = data_woodruff_preprocessed.nlargest(3, 'percentage_match')[['phrase', 'scripture_text', 'percentage_match']]
        results = pd.concat([results, top_3_rows])

print(results)

