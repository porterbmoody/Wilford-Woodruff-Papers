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

