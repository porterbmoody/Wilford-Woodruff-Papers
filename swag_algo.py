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
import random

class NeuralNetwork:
    
    def __init__(self, inputs, outputs) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.weights = [random.random() for i in range(4)]

    def train(self):
        print('training')
        hidden_layer = NeuralNetwork.dot(self.inputs, self.weights)
        print(hidden_layer)

    def update_weights(self):
        for weight in self.weights:
            continue

    def dot(v1, v2):
        y = []
        for x1, x2 in zip(v1, v2):
            y.append(x1*x2)
        return y

    def display(self):
        print(self.inputs)
        print(self.outputs)
        print(self.weights)

inputs = [1,2,3,4,5]
outputs = [1,0,1,0,0]

neural_network = NeuralNetwork(inputs, outputs)
# neural_network.display()
neural_network.train()


