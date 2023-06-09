import pandas as pd
from DataUtil import DataUtil


class ScriptureData:


    def __init__(self, path):
        self.data_scriptures_raw = pd.read_csv(path)

    def clean_data(self):
        print('cleaning scripture data...')
        # books = ["New Testament", "Book of Mormon", "Doctrine and Covenants", "Pearl of Great Price"]
        books = ['Doctrine and Covenants']
        # filter down to only certain books
        self.data_scriptures = self.data_scriptures_raw[self.data_scriptures_raw['volume_title'].isin(books)]
        self.data_scriptures['scripture_text'] = self.data_scriptures['scripture_text'].str.lower()

    def preprocess_data(self):

        self.data_scriptures = self.data_scriptures[['volume_title', 'verse_title', 'scripture_text']]
        self.data_scriptures['scripture_text'] = self.data_scriptures['scripture_text'].apply(DataUtil.str_replace_list, regex_list = stop_words, replacement = ' ')
        self.data_scriptures