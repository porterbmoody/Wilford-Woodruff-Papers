import pandas as pd
from DataUtil import DataUtil


class WoodruffData:


    typos = {
        r'sacrafice'    : r'sacrifice',
        r'discours'     : r'discourse',
        r'travling'      : r'traveling',
        r'oclock'       : r'oclock',
        r'[Ww]\. [Ww]oodruff' : r'Wilford Woodruff',
        r'any\s?whare'    : r'anywhere',
        r'some\s?whare'     : r'somewhere',
        r'whare'         : r'where',
        r'sumthing'      : r'something',
        r' els '         : r' else ',
        r'savio saviour' : r'saviour',
        r'intirely    ' : r'entirely',
        r'phylosophers' : r'philosophers',
        r'baptised'     : r'baptized',
        r'benef\- it'   : r'benefit',
        r'preachi \-ng'      : r'preaching',
        r'oppor- tunities' : r'opportunities',
        r'vary'         : r'very',
        r'Councellor'   : r'Counselor',
        r'councellor'   : r'counselor',
        r'sircumstances' : r'circumstances',
        r'Preasent'    : r'present',
        r'Sept\.'      : r'September',
        r'Sacramento Sacramento' : r'Sacramento',
        r'tryed'       : r'tried',
        r'fals'        : r'false',
        r'Aprail'      : r'April',
        r'untill'      : r'until',
        }

    symbols = {
            # 'b\. 1795':'',
            r'(, b\.)'              : r'',
            r'\<U\+25CA\>'          : r'',
            r'\&amp;c?'             : r"and",
            r'\&apos;'              : r"'",
            r"(\^?FIGURES?\^?)"     : r'',
            r'[\{\}\~]'             : r'',
            r'\s{2}'                : r' ',
            r','                    : r'',
            r'\[(\w+)\]'            : r'',
            r'\n'                   : r' ',
            r'\[\[(.*?)\|(.*?)\]\]' : r'\1',
            r'\- '                  : r'',
            r'- ng '                 : r'ng ',
            r' ng '                 : r'ng ',
            r' ed '                 : r'ed ',
            r'\n'                 : r' ',
            r'\s+'                 : r' ',
            r'\.'                 : r'',
        }

    entries_to_remove = [
        r'WW 1841-2',
        r'Front cover',
        r'THE SECOND BOOK OF WILLFORD FOR 1839',
        r'W\. WOODRUFFs DAILY JOURNAL AND HISTORY IN 1842',
        r"WILFORD WOODRUFF's DAILY JOURNAL AND HISTORY IN 1843",
        r"WILLFORD WOODRUFF'S JOURNAL VOL\. 2\. AND A SYNOPSIS OF VOL\. 1\.",
        r"Willford Woodruff's Journal Containing an Account Of my life and travels from the time of my first connextion with the Church of Jesus Christ of Latter-day Saints",
        r'THE FIRST BOOK OF WILLFORD VOL\. 2\. FOR 1838',
        r"WILLFORD\. WOORUFF's DAILY JOURNAL AND TRAVELS",
        r'Pgs 172\–288 are blank',
        ]

    def __init__(self, path) -> None:
        self.data_woodruff_raw = pd.read_csv(path).query("`Document Type` == 'Journals'")


    def clean_data(self):
        ## data cleaning
        # rename text column to just 'text'
        self.data_woodruff = self.data_woodruff_raw.rename(columns={"Text Only Transcript": "text"})

        columns = ['Document Type', 'Parent Name', 'text']
        self.data_woodruff = self.data_woodruff[columns]


        self.data_woodruff['text'] = self.data_woodruff['text'].replace(self.typos, regex=True)
        self.data_woodruff['text'] = self.data_woodruff['text'].replace(self.symbols, regex=True)

        # loop through entries and remove rows that have regex match in entry
        for entry in self.entries_to_remove:
            data_woodruff = DataUtil.regex_filter(data_woodruff, 'text', entry)
            # data_woodruff[data_woodruff['text'].str.contains(entry) == False]


    def preprocess_data(self):
        ## data preprocessing

        # """ Preprocess the pandas dataframe for model training.
        # This entails selecting only relevant columns,
        # then expanding the text column into phrases of size n (in this case 15)
        # First is separates each entry into a list where each element contains a 15 word string.
        # Then it explodes the dataframe so that each element of the list is mapped to its own row
        # So that each row contains a single 15 word phrase of an entry
        # """
        self.data_woodruff_preprocessed = self.data_woodruff

        # remove stopwords
        stop_words = [' and ', r' the', ' that ',
                    ' of ', ' to ',
                    ' with ', ' at ', ' by ', ' in ',
                    ' on ',
                    ' for ',
                    ' us ',
                    ' we ',
                    ' my ',
                    ' his ',
                    r' \. ',
                    r' i '
                    r' \, ']

        self.data_woodruff_preprocessed['text'] = self.data_woodruff_preprocessed['text'].apply(DataUtil.str_replace_list, regex_list = stop_words, replacement = ' ')
        # lowercase all text
        self.data_woodruff_preprocessed['text'] = self.data_woodruff_preprocessed['text'].str.lower()

        self.data_woodruff_preprocessed['phrase'] = self.data_woodruff_preprocessed['text'].apply(DataUtil.split_string_into_list, n = 15)
        self.data_woodruff_preprocessed = self.data_woodruff_preprocessed.explode('phrase')

        # count number of words in 'text'
        self.data_woodruff_preprocessed['word_count'] = self.data_woodruff_preprocessed['phrase'].apply(DataUtil.count_words)

        # we'll just remove all these cuz they're weird
        self.data_woodruff_preprocessed.query('word_count < 5').head(100)

        self.data_woodruff_preprocessed = self.data_woodruff_preprocessed.query('word_count > 5')
