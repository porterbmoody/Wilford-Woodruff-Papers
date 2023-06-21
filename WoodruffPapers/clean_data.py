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


#%%
woodruff_data.clean_data()
# woodruff_data.data = woodruff_data.data.head(1)

woodruff_data.preprocess_data()

#%%

text_sample = DataUtil.combine_rows(woodruff_data.data['text'].head(10))
DataUtil.create_frequency_dist(text_sample).head(100)

#%%
# read scripture data
url_scriptures = 'https://github.com/wilfordwoodruff/wilford_woodruff_hack23/raw/main/data/lds-scriptures.csv'
path_scriptures = '../data/scriptures.csv'
scripture_data = ScriptureData(path_scriptures)

scripture_data.clean_data()
scripture_data.preprocess_data()


scripture_data.data.head(100)
text_sample = DataUtil.combine_rows(scripture_data.data['scripture_text'].head(10))
DataUtil.create_frequency_dist(text_sample).head(100)


#%%

## run verse matching algorithm
data_sample = woodruff_data.data_preprocessed.head(600)

results = pd.DataFrame()

for index, row in scripture_data.data.iloc[300 : 400].iterrows():
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
        top_3_rows = data_sample.nlargest(3, 'percentage_match')[['phrase', 'scripture_text', 'percentage_match', 'text']]
        top_3_rows['verse_title'] = verse_title
        # add top 3 percentage match rows to results dataframe
        results = pd.concat([results, top_3_rows])


results.sort_values(by = 'percentage_match', ascending=False)



# %%




# %%

