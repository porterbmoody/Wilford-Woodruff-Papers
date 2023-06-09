from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from DataUtil import DataUtil
import numpy as np


class AISwag:

    @staticmethod
    def compute_match_percentage(text_woodruff, scripture_text):
        words_woodruff = DataUtil.str_split(text_woodruff)
        words_scripture = DataUtil.str_split(scripture_text)
        vectorizer = TfidfVectorizer()
        ### vectorize words
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