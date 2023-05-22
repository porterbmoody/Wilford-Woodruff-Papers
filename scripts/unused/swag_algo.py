#%%
import tensorflow_hub as hub


embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
embed
#%%

text = 'heres a body of really awesome text. bro check this cool swag sentance its awesome and talks about wilford woodruff breh. maybe even a biography of wilford woodruff. Heres a second sentance with some super awesome cool stuff bro.'
woodruff = 'heres some really awesome and cool text of a ton of journal entries from wilford woodruff!! '

sentences = text.split('.')
sentences



#%%
# embed sentences
sentence_vectors = embed(sentences)
sentence_vectors

#%%

for sentence, sentence_vector in zip(sentences, sentence_vectors):
    print(sentence)
    print(sentence_vector)

encoding_mappings = {
    'god|lord' : 1,
    'prophet' : .9,
    'revelation' : .9,
    'holy ghost' : 1,
    'spirit' : 1,
    'prophecy' : 1,
    'gentile':.8,
    'jesus' : 1,
    'christ' : 1,
    'covenants' : 1,
    'blessing' : 1,
    'Melchizedek' : 1,
    'church of jesus christ' : 1,
    'church of jesus christ of latter day saints' : 1,
    'saints' : 1,
    'Book of mormon' : 1,
    'fullness of the everlasting gospel' : 1,
    'fullness of the gospel' : 1,
    'gospel' : 1,
    'Abraham' : 1,
}


#%%

class SentenceAlgo():
    BETWEEN_SENTENCES = r'\(. )'
    BETWEEN_WORDS = r' '

    def __init__(self) -> None:
        self.score = 0

    def compute_similarity(self, sentence1, sentence2):
        sentence1_words = sentence1.split(SentenceAlgo.BETWEEN_WORDS)
        sentence2_words = sentence2.split(SentenceAlgo.BETWEEN_WORDS)
        print(sentence1_words)
        print(sentence2_words)
        for word1 in sentence1_words:
            for word2 in sentence2_words:
                self.score += SentenceAlgo.compare_words(word1,word2)

    def vectorize(words):
        sentence_vector = {}
        id = 0
        for word in sentence1:
            if id not in sentence_vector.values():
                sentence_vector[word] = id
                id += 1

    @staticmethod
    def pre_process_word(word):
        return word.lower()

    @staticmethod
    def compare_words(word1, word2):
        word1, word2 = SentenceAlgo.pre_process_word(word1),SentenceAlgo.pre_process_word(word2)
        if word1 == word2:
            return 1
        else:
            return 0
    
    def display_score(self):
        print('sentence score:', self.score)
    # @staticmethod
    # def word_instances_in_single_sentence(word):
        # words = sentence.split(SentenceAlgo.BETWEEN_WORDS)

sentence1 = 'hello i am super cool and awesome sentence'
sentence2 = 'hello i am another swaggy sentence bro read me if you want to read a very cool sentence'

sentence_algo = SentenceAlgo()
sentence_algo.compute_similarity(sentence1, sentence2)
sentence_algo.display_score()


# %%

import numpy as np

class NeuralNetwork:

    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.hidden_layer_size = 2
        self.weights1 = np.random.rand(self.hidden_layer_size, len(self.inputs))
        self.weights2 = np.random.rand(len(self.outputs), self.hidden_layer_size)

    def feed_forward(self):
        print('Feeding Forward:')
        print('computing hidden layer...')
        print('W.x:', self.weights1, self.inputs)
        self.hidden_layer = np.dot(self.weights1, self.inputs)
        print(self.hidden_layer)
        print()
        print('computing predictions...')
        self.predictions = np.dot(self.weights2, self.hidden_layer)
        print('Predictions:', self.predictions)
        loss = self.compute_loss()
        print(loss)

    def back_propagate(self):
        print('updating weights1...')
        print(self.weights1)
        rows, cols = self.weights1.shape
        for i in range(rows):
            for j in range(cols):
                element = self.weights1[i, j]
                print(element)

        # errors = self.outputs - self.predictions
        # gradients = errors * self.inputs
        # self.weights1 += gradients

    def compute_loss(self):
        errors = np.array((self.outputs - self.predictions) ** 2)
        print('errors', errors)
        return np.sum(errors)

    def display(self):
        print('Inputs:', self.inputs)
        print('Outputs:', self.outputs)
        print('Weights1:', self.weights1)
        print('Weights2:', self.weights2)
        # print('Loss:', self.compute_loss())

inputs = [1, 2, 3, 1, 2]
outputs = [1, 0, 1, 0, 0]

neural_network = NeuralNetwork(inputs, outputs)
neural_network.feed_forward()
neural_network.back_propagate()
# neural_network.display()


#%%

import pandas as pd
from termcolor import colored
import altair as alt


scriptures = 'C:/Users/porte/Desktop/coding/hackathon23_winner/data/lds-scriptures.csv'
data_scriptures = pd.read_csv(scriptures)

# wwp_raw_url = 'https://github.com/wilfordwoodruff/DSS-W23-Project/blob/master/raw_data/wwp.csv'
# wwp_journals = 'https://github.com/wilfordwoodruff/hackathon23_winner/blob/main/data/journals.csv'
# journal_entries_from_1836_to_1895 = 'https://raw.githubusercontent.com/wilfordwoodruff/Consult_S23_WWP/master/data/derived/papers.csv?token=GHSAT0AAAAAACB5DCILP5SNGHVTVWFZJALWZC36VIA'
raw_entries_clean = 'data/raw_entries_clean.csv'
data_journals = pd.read_csv(raw_entries_clean)

data_journals

#%%
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

model_path = "models/word2vec.model"
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save(model_path)


#%%
model = Word2Vec.load(model_path)
model.train([["lds", "church"]], total_examples=1, epochs=1)
model

#%%
vector = model.wv['mormon']  # get numpy vector of a word
vector

#%%
sims = model.wv.most_similar('church', topn=10)
sims

# %%
import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers