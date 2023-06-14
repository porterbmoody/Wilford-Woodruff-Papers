#%%
from DataUtility import DataUtility

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

## old EDA
#%%
import pandas as pd
from termcolor import colored
import altair as alt
import re
import nltk

scriptures = '../data/lds-scriptures.csv'
data_scriptures = pd.read_csv(scriptures)

# wwp_raw_url = 'https://github.com/wilfordwoodruff/DSS-W23-Project/blob/master/raw_data/wwp.csv'
# wwp_journals = 'https://github.com/wilfordwoodruff/hackathon23_winner/blob/main/data/journals.csv'
# journal_entries_from_1836_to_1895 = 'https://raw.githubusercontent.com/wilfordwoodruff/Consult_S23_WWP/master/data/derived/papers.csv?token=GHSAT0AAAAAACB5DCILP5SNGHVTVWFZJALWZC36VIA'
raw_entries_clean = '../data/raw_entries_clean.csv'
data_journals = pd.read_csv(raw_entries_clean)

data_scriptures
data_journals


#%%
data_journals['date'] = pd.to_datetime(data_journals['date'])
data_journals

#%%
data_journals_grouped_1840 = (data_journals
                         .query('year == 1840')
                         .groupby('month').agg(sum).reset_index())
data_journals_grouped_1840


# 1840 word count
alt.Chart(data_journals_grouped_1840, title = 'Wilford Woodruff year 1840').encode(
    x = 'month',
    y = 'word_count'
).mark_line()
#%%
alt.Chart(data_journals_grouped_1840, title = 'Wilford Woodruff year 1847').encode(
    x = 'month',
    y = 'word_count'
).mark_line()
#%%
# 1847 word count
data_journals_grouped_1847 = (data_journals
                         .query('year == 1847')
                         .groupby('month').agg(sum).reset_index())
data_journals_grouped_1847

#%%
# by month
alt.Chart(data_journals_grouped_1847, title = 'Wilford Woodruff year 1847').encode(
    x = 'month',
    y = 'word_count'
).mark_line()
#%%
# by year
alt.Chart(data_journals.query('year == 1847 & month==4'), title = 'Wilford Woodruff year 1847').encode(
    x = 'date',
    y = 'word_count'
).mark_line()



#%%

import pandas as pd
from DataUtil import DataUtil

data = pd.DataFrame({
    'col1' : ['gathering of israel','book of mormon pizza pizza','god lord jesus christ pizza'],
    'First Date': ['Journal (December 29, 1833 – January 3, 1838)',
                   'Journal (December 29, 1833 – January 3, 1838)',
                   'Journal (January 1893 – April 1897)']
})

word = 'pizza'

date_regex = r"\b\w+\s\d{4}\b"
date_regex = r"\w+?\s?\d{1,2}?,?\s?\d{4}"
date_regex = r"1[98]\d{2}"


## bro if its working here why aint it working over there
data['year'] = data['First Date'].apply(DataUtil.str_extract, regex = date_regex)

data

# DataUtil.str_extract('Journal (December 29, 1833 – January 3, 1838)', date_regex)

#%%
import pandas as pd
from WoodruffData import WoodruffData
import altair as alt
from DataUtil import DataUtil

# read woodruff data
url_woodruff = "https://github.com/wilfordwoodruff/Main-Data/raw/main/data/derived/derived_data.csv"
path_woodruff = '../data/data_woodruff_raw.csv'
path_woodruff = '../data/raw_entries.csv'
woodruff_data = WoodruffData(path_woodruff)


woodruff_data.clean_data()

woodruff_data.data



#%%
word = 'book of mormon'
woodruff_data.data['count_'+word] = woodruff_data.data['text'].apply(DataUtil.str_count_occurrences,
                                                                       word=word)

woodruff_data.data


data = woodruff_data.data.groupby(['year', 'count_' + word]).agg(sum).reset_index()


# group by every 5 years
data.groupby(data['year'])['count_'+word].sum()


#%%


# woodruff_data.data.value_counts('count_'+word)

alt.Chart(data, title = 'mentions of the word '+word).encode(
    x ='year',
    y = 'count_'+word,
).mark_boxplot()


# %%


string = "Journal (December 29, 1833 – January 3, 1838)"

date_regex = r"\b\w+\s\d{1,2},\s\d{4}\b"


DataUtil.str_extract(string, regex = date_regex)


