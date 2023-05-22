#%%
import pandas as pd
from termcolor import colored
import altair as alt
import re

scriptures = '../data/lds-scriptures.csv'
raw_entries_clean = '../data/raw_entries_clean.csv'

data_scriptures = pd.read_csv(scriptures)
data_journals = pd.read_csv(raw_entries_clean)

data_clean = data_journals.copy()
data_sample = data_journals.head(50)
data_scriptures
data_journals

#%%
# loading all entries into one string
# text_sample = ''
# for index, row in data_clean.head(10).iterrows():
#     text_sample += row['text']
# text_sample

text = ''
for index, row in data_clean.iterrows():
    text += row['text']
text
#%%
# pattern = r'\[\[(.*?)\]\]'
pattern = r'\&amp;'
len(re.findall(pattern, text))

#%%
class WWPDataCleaning:
    # first clean all the '\n' new line places
    @staticmethod
    def remove_new_lines(string):
        return re.sub(r'\n', r' ', string)

    # clean suggestions
    @staticmethod
    def clean_suggestions(string):
        pattern = r'\[\[(.*?)\]\]'
        suggestions = re.findall(pattern, string)
        for suggestion in suggestions:
            replacement = suggestion.split('|')[0]
            suggestion_exact_match = '[[' + suggestion + ']]'
            string = string.replace(suggestion_exact_match, replacement)
        return string

    @staticmethod
    def clean_ampersands(string):
        return re.sub(r'\&amp;', r'and', string)
string = 'I picked Peaches &amp; Plums &amp; hoed in the garden'

WWPDataCleaning.clean_ampersands(string)

#%%


data_clean['text'] = data_clean['text'].apply(WWPDataCleaning.remove_new_lines)
data_clean['text'] = data_clean['text'].apply(WWPDataCleaning.clean_suggestions)
data_clean['text'] = data_clean['text'].apply(WWPDataCleaning.clean_ampersands)
data_clean.to_csv('../data/journal_entries_clean.csv')


#%%
data_sample['text'] = data_sample['text'].apply(clean_suggestions)
data_sample

# data_clean['text'] = data_clean['text'].apply(clean_suggestions)
# data_clean
# data_clean.to_csv('../data/journal_entries_clean.csv')

#%%


#%%
# data_journals['book_of_mormon_frequency'] = data_journals['text'].str.count('book of mormon')

# (data_journals['text'].str.count('book of mormon').
#  .sort_values(by = 'book_of_mormon_frequency', ascending=False)
#  .groupby('book_of_mormon_frequency').count())

# #%%


# data1 = data_journals
# # data1['text'].apply(' '.join).reset_index()
# # data1
# text = ''
# for index, row in data1.head(3).iterrows():
#     text += row['text']
#     # words = re.split(r' ', text)
#     # print(words)
# print(text)

# #%%
# data1['text'].str.contains('prophet', case=False, regex = True).value_counts()
# data1['text'].str.contains('(book of mormon)', case=False, regex = True).value_counts()
# import re
# import pandas as pd

# def clean_suggestions(string):
#     pattern = r'\[\[(.*?)\]\]'
#     suggestions = re.findall(pattern, string)
#     for suggestion in suggestions:
#         replacement = suggestion.split('|')[0]
#         suggestion = '[[' + suggestion + ']]'
#         string = string.replace(suggestion, replacement)
#     return string

# # Create a sample DataFrame
# data = {'text': ['sdaf yo bruh [[bro|b]] sdvsaf', '[[hello|greeting]] world', 'some [[text|words]]']}
# df = pd.DataFrame(data)
# print(df)

# # Apply the clean_suggestions function to each cell in the 'text' column
# df['text'] = df['text'].apply(clean_suggestions)

# # Print the modified DataFrame
# df
