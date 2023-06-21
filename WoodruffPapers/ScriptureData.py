import pandas as pd
from DataUtil import DataUtil


class ScriptureData:


    def __init__(self, path):
        self.data_raw = pd.read_csv(path)

    def clean_data(self):
        print('cleaning scripture data...')
        books = ["Book of Mormon", "Doctrine and Covenants", "Pearl of Great Price"]#"New Testament",
        # books = ['Doctrine and Covenants']
        # filter down to only certain books
        self.data = self.data_raw[self.data_raw['volume_title'].isin(books)]
        self.data['scripture_text'] = self.data['scripture_text'].str.lower()

        # select only relavant columns
        self.data = self.data[['volume_title', 'verse_title', 'scripture_text']]

    def preprocess_data(self):
        # remove stopwords
        self.data['scripture_text'] = self.data['scripture_text'].replace(DataUtil.stop_words, regex=True)
        # self.data['scripture_text'] = self.data['scripture_text'].apply(DataUtil.remove_stop_words)