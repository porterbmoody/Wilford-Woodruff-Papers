#%%
import pandas as pd
import numpy as np
import plotly.express as px
from termcolor import colored
from WoodruffPatterns import typos, symbols, entries_to_remove
from MLModel import MLModel
from StrUtil import StrUtil


pd.set_option('display.max_rows', 100)
pd.options.display.max_colwidth = 200
pd.options.mode.chained_assignment = None
# this powers the word_tokenize function
# nltk.download('punkt')


#%%
# read in data
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
path_woodruff = '../data/data_woodruff.csv'
path_woodruff1 = '../data/data_woodruff1.csv'
data_woodruff = pd.read_csv(path_woodruff).query("`Document Type` == 'Journals'")

data_woodruff = data_woodruff.rename(columns={"Text Only Transcript": "text"})

columns = ['Parent Name', 'text', 'People']
data_woodruff = data_woodruff[columns]
data_woodruff
# woodruff_papers.data_woodruff[columns].to_csv(path_woodruff1,  index = False)

#%%

## data cleaning
# text_sample = StrUtil.combine_rows(data_woodruff.iloc[200:210]['text'])
# text_sample

def clean_string(string):
    # put cleaning into 1 function so we can apply it to each row...
    # text_sample = text_sample.lower()

    string = StrUtil.replace_with_dict(string, regex_dict = typos)
    string = StrUtil.replace_with_dict(string, regex_dict = symbols)
    return string

data_woodruff['text'] = data_woodruff['text'].apply(clean_string)

for entry in entries_to_remove:
    StrUtil.regex_filter(data_woodruff, 'text', entry)

StrUtil.create_frequency_dist(string = data_woodruff['text'][0]).head(100)


#%%
StrUtil.create_frequency_dist(string = data_woodruff['text'][0]).tail(100)


#%%
# read scriptures

books = ["New Testament", "Book of Mormon", "Doctrine and Covenants", "Pearl of Great Price"]

url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
path_scriptures = '../data/scriptures.csv'
# pd.read_csv(path_scriptures).to_csv(path_scriptures, index = False)

data_scriptures = pd.read_csv(path_scriptures)

data_scriptures = data_scriptures[data_scriptures['volume_title'].isin(books)]
data_scriptures['scripture_text'] = data_scriptures['scripture_text'].str.lower()

data_scriptures



#%%

## data preprocessing

# """ Preprocess the pandas dataframe for model training.
# This entails selecting only relevant columns,
# then expanding the text column into phrases of size n (in this case 15)
# First is separates each entry into a list where each element contains a 15 word string.
# Then it explodes the dataframe so that each element of the list is mapped to its own row
# So that each row contains a single 15 word phrase of an entry
# """

common_words = [' and ', ' the ', ' of ', ' to ',
                ' with ', ' at ', ' by ', ' in ',
                r'\.',
                r'i']
text_sample = StrUtil.str_replace_list(text_sample,
                                       regex_list = common_words,
                                       replacement=' ')


data_woodruff['phrase'] = data_woodruff['text'].apply(StrUtil.split_string_into_list, n = 15)
data_woodruff = data_woodruff.explode('phrase')

data_woodruff

#%%

## train model
results = pd.DataFrame()
for index, row in scriptures.data_scriptures.iloc[300:350].iterrows():
    verse_title    = row['verse_title']
    scripture_text = row['scripture_text']
    print(verse_title)
    print(scripture_text)

    woodruff_papers.data_woodruff['scripture_text'] = scripture_text
    woodruff_papers.data_woodruff['percentage_match'] = (woodruff_papers.data_woodruff['phrase'].apply(MLModel.compute_match_percentage, scripture_text=scripture_text))
    woodruff_papers.data_woodruff.sort_values(by = 'percentage_match', ascending=False)

    if woodruff_papers.data_woodruff['percentage_match'].max() > 50:
        print(colored('attaching results', 'green'), woodruff_papers.data_woodruff['percentage_match'].max())
        top_3_rows = woodruff_papers.data_woodruff.nlargest(3, 'percentage_match')[['phrase', 'scripture_text', 'percentage_match']]
        results = pd.concat([results, top_3_rows])

# %%


