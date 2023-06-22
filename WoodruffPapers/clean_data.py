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


woodruff_data.data_raw
# woodruff_data.data_raw.to_csv(path_woodruff)

woodruff_data.clean_data()
woodruff_data.data


# woodruff_data.data = woodruff_data.data.head(1)
# woodruff_data.preprocess_data()

#%%

s

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