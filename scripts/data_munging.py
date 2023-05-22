#%%
import pandas as pd
from termcolor import colored
import altair as alt
import re


scriptures = 'C:/Users/porte/Desktop/coding/hackathon23_winner/data/lds-scriptures.csv'
data_scriptures = pd.read_csv(scriptures)

# wwp_raw_url = 'https://github.com/wilfordwoodruff/DSS-W23-Project/blob/master/raw_data/wwp.csv'
# wwp_journals = 'https://github.com/wilfordwoodruff/hackathon23_winner/blob/main/data/journals.csv'
# journal_entries_from_1836_to_1895 = 'https://raw.githubusercontent.com/wilfordwoodruff/Consult_S23_WWP/master/data/derived/papers.csv?token=GHSAT0AAAAAACB5DCILP5SNGHVTVWFZJALWZC36VIA'
raw_entries_clean = 'data/raw_entries_clean.csv'
data_journals = pd.read_csv(raw_entries_clean)

data_scriptures
data_journals
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