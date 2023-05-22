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


#%%
#%%
# def count_word_occurences(word):

series[word] = data_journals['text'].str.count('prophet')
series.groupby(word)

data_journals = count_word_occurences('prophet')
data_journals

#%%
data_journals['book_of_mormon_frequency'] = data_journals['text'].str.count('book of mormon')

(data_journals['text'].str.count('book of mormon').
 .sort_values(by = 'book_of_mormon_frequency', ascending=False)
 .groupby('book_of_mormon_frequency').count())

#%%


data1 = data_journals
# data1['text'].apply(' '.join).reset_index()
# data1
text = ''
for index, row in data1.head(3).iterrows():
    text += row['text']
    # words = re.split(r' ', text)
    # print(words)
print(text)

#%%
data1['text'].str.contains('prophet', case=False, regex = True).value_counts()
data1['text'].str.contains('(book of mormon)', case=False, regex = True).value_counts()


#%%

def clean_suggestions(string):
    """
    This function takes in a string and finds all occurences of '[[some words|some other words]]'.
    Then it does an exact match replacement in the passed in string of the words to the left of the '|'
    input: str
    output: str

    for example: 
    if the input string = 'sdaf yo bruh homie dude bruh [[bro|b]] sdvsaf  sadf sdfaasdf as [[some words|asf]]'
    the output string will find the words 'bro' and 'some words' and replace them and return the string
    sdaf yo bruh homie dude bruh bro sdvsaf  sadf sdfaasdf as some words
    """
    pattern = r'\[\[(.*?)\]\]'
    suggestions = re.findall(pattern, string)
    print('suggestions found', len(suggestions))
    for suggestion in suggestions:
        # get characters to the left of |
        replacement = suggestion.split('|')[0]
        # add the [[ and ]] back in to do an exact match replacement on [[some words|some other words]]
        suggestion = '[[' + suggestion + ']]'
        print('suggestion:', suggestion)
        print('replacement:', replacement)
        print('replacing...')
        string = string.replace(suggestion, replacement)

    return string

string = 'sdaf yo bruh homie dude bruh [[bro|b]] sdvsaf  sadf sdfaasdf as [[some words|asf]]'
print(string)
print()
clean_string = clean_suggestions(string)
print()
print(clean_string)

#%%

clean_suggestions(text)