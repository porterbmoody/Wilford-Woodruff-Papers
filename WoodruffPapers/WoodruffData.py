import pandas as pd
from DataUtil import DataUtil
from datetime import datetime


class WoodruffData:

    religious_words = ['joseph smith jr',
             'eternity',
             'book of mormon',
             'heaven',
             'brigham young',
             'priesthood',
             'babylon',
             'bible',
             'new testament',
             'commandment',
             'gospel',
             'new testament',
             'dispensation',
             'commanded',
            'god',
            'lord',
            'book',
            'hearken',
            'inhabitants',
            'man',
            'earth',
            'baptized',
            'baptism',
            'alexander',
            'monday',
            'kentucky',
            'county',
            'day',
            'faith',
            'bless',
            'blessing',
            'blessed',
            'jesus christ',
            'christ',
            'jesus',
            'elder',
            'elders',
            'ordinances',
            'temples',
            'house of the lord',
            'abide',
            'repentance',
            'repent',
            'repents',
            'holy ghost',
            'spirit',
            'miracle',
            'miracles',
            'spoken',
            'spoke',
            'spake',
            'kingdom',
            'witness',
            'strength',
            'celestial',
            'power',
            'earnest',
            'preach the gospel',
            'preach',
            'inspired',
            'satan',
            'console',
            'account',
            'sunday',
            'church',
            'restored church',
            'church of jesus christ of latter day saints',
            'salvation',
            'lamanites',
            'nephites',
            'lehi',
            'nephi',
            'righteous',
            'holy',
            'daniel',
            'worthy',
            'prophecy',
            'ohio',
            'revelations',
            'savior',
            '2nd coming',
            'glory',
            'true',
            'chaste',
            'god rules',
            ]

    def __init__(self, path) -> None:
        self.data_raw = pd.read_csv(path)
        if "`Document Type`" in self.data_raw.columns:
            self.data_raw = self.data_raw.query("`Document Type` == 'Journals'")

    def clean_data(self):
        ## data cleaning
        # rename text column to just 'text'
        self.data = self.data_raw
        # self.data = self.data.rename(columns={"Text Only Transcript": "text"})

        # self.data['text'] = self.data['Text Only Transcript']
        # columns = ['Document Type', 'Parent Name','Name',  'text','Text Only Transcript']

        # self.data = self.data.dropna(subset=['date'])

        # date_format = "%B %d, %Y"
        # self.data['date'] = pd.to_datetime(self.data['date'], format=date_format)
            # data[data['text'].str.contains(entry) == False]

        # for url
        # regex_year =r'\d{4}'
        # self.data['year'] = self.data['date'].apply(DataUtil.str_extract, regex=regex_year)

        # for raw_entries
        # date_regex = r"\w+\s\d{1,2}\,\s\d{4}|\w+\s\d{4}"
        # self.data['date'] = self.data['Parent Name'].apply(DataUtil.str_extract, regex = date_regex)


    def preprocess_data(self):
        ## data preprocessing

        # """ Preprocess the pandas dataframe for model training.
        # This entails selecting only relevant columns,
        # then expanding the text column into phrases of size n (in this case 15)
        # First is separates each entry into a list where each element contains a 15 word string.
        # Then it explodes the dataframe so that each element of the list is mapped to its own row
        # So that each row contains a single 15 word phrase of an entry
        # """
        self.data_preprocessed = self.data



        # remove stopwords
        self.data_preprocessed['text'] = self.data_preprocessed['text'].replace(DataUtil.stop_words, regex=True)

        # explode each entry into a separate row of 15 words
        self.data_preprocessed['phrase'] = self.data_preprocessed['text'].apply(DataUtil.split_string_into_list, n = 15)
        self.data_preprocessed = self.data_preprocessed.explode('phrase')
        self.data_preprocessed = self.data_preprocessed.dropna(subset=['phrase'])
        print(self.data_preprocessed['phrase'])
        # for pizza in list(self.data_preprocessed['phrase']):
            # print(pizza)
        # print(self.data_preprocessed.info())
        # count number of words in 'text'
        self.data_preprocessed['word_count'] = self.data_preprocessed['phrase'].apply(DataUtil.count_words)

        # we'll just remove all phrases with less than 5 words cuz they're weird
        self.data_preprocessed.query('word_count < 5').head(100)

        self.data_preprocessed = self.data_preprocessed.query('word_count > 5')
