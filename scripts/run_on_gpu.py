#%%
from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from DataUtil import DataUtil

# function optimized to run on gpu
@jit(target_backend='cuda')
def extract_matches(string):
	# for i in range(100000000):
		# a[i]+= 1
	for element in string.split():
		element += ' hello'
		print(element)

if __name__=="__main__":

	start = timer()
	string = 'hello my name is porter'
	extract_matches(string)
	print("with GPU:", timer()-start)


#%%
def extract_matches(phrases_woodruff, tfidf_matrix_woodruff, vectorizer, phrases_scriptures):

        tfidf_matrix_scriptures = vectorizer.transform(phrases_scriptures)

        similarity_matrix = cosine_similarity(tfidf_matrix_woodruff, tfidf_matrix_scriptures)
        # time.sleep(1)
        threshold = 0.65  # Adjust this threshold based on your preference
        similarity_scores = []
        top_phrases_woodruff = []
        top_phrases_scriptures = []

        for i, phrase_woodruff in enumerate(phrases_woodruff):
            for j, phrase_scriptures in enumerate(phrases_scriptures):
                similarity_score = similarity_matrix[i][j]
                # print(similarity_score)
                # print(phrase_scriptures)
                # print(phrase_woodruff)
                if similarity_score > threshold:
                    top_phrases_woodruff.append(phrase_woodruff)
                    top_phrases_scriptures.append(phrase_scriptures)
                    similarity_scores.append(similarity_score)

        data = pd.DataFrame({
             'phrase_woodruff':top_phrases_woodruff,
             'phrases_scriptures':top_phrases_scriptures,
             'similarity_scores' : similarity_scores}).sort_values(by='similarity_scores',ascending=False)
        return data

#%%


import numpy as np
import numba
from numba import cuda

print(np.__version__)
print(numba.__version__)

cuda.detect()