from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from DataUtil import DataUtil
import numpy as np


class AISwag:
    key_words = ['joseph smith jr',
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

    @staticmethod
    def compute_match_percentage(text_woodruff, scripture_text):
        words_woodruff = DataUtil.str_split(text_woodruff)
        words_scripture = DataUtil.str_split(scripture_text)
        # print(words_woodruff)
        woodruff_score = 0
        # loop through scripture words to see if it has any key words
        # for word_woodruff in words_woodruff:
        #     if word_woodruff in AISwag.key_words and word_woodruff in words_scripture:
        #         woodruff_score += 1
                # print(woodruff_score)
                # loop through woodruff word to see if it also contains matches
                # for word_woodruff in words_woodruff:
                    # print('|' + str(word_woodruff) + '|')
                    # print('word in list:', str(word_woodruff in AISwag.key_words))
                    # if word_woodruff in AISwag.key_words:
                        # print('p' + word_woodruff + 'p')
                        # if it matches add 1 to the score
        percent_match_woodruff = woodruff_score / len(words_woodruff)
        # print(percent_match_woodruff)

        ## we'll try something by hand for a moment
        ### vectorize words
        vectorizer = TfidfVectorizer()
        # Compute TF-IDF matrices
        tfidf_matrix_woodruff = vectorizer.fit_transform(words_woodruff)
        tfidf_matrix_verse = vectorizer.transform(words_scripture)

        similarity_scores = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_verse)
        vectorizer.get_feature_names_out()
        woodruff_word_match_ids   = np.unique(np.where(similarity_scores == 1)[0])
        scriptures_word_match_ids = np.unique(np.where(similarity_scores == 1)[1])
        percent_match_woodruff    = round(len(woodruff_word_match_ids) / len(words_woodruff), 4)
        percent_match_scripture    = round(len(scriptures_word_match_ids) / len(words_scripture), 4)
        return percent_match_woodruff