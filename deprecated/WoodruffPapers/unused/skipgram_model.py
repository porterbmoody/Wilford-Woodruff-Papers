#%%
import io
import re
import string
import tqdm

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

#%%
sentence = "Conference Met at 10 oclok. President Joseph Young prayed George Albert Smith spoke "
tokens = list(sentence.lower().split())
print(len(tokens))
print(tokens)


#%%

vocab, index = {}, 1  # start indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in tokens:
  if token not in vocab:
    vocab[token] = index
    index += 1
vocab_size = len(vocab)
print(vocab)

inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)

#%%
example_sequence = [vocab[word] for word in tokens]
print(example_sequence)

window_size = 2
positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
      example_sequence,
      vocabulary_size=vocab_size,
      window_size=window_size,
      negative_samples=0)
print(len(positive_skip_grams))
# %%
for target, context in positive_skip_grams[:5]:
  print(f"({target}, {context}): ({inverse_vocab[target]}, {inverse_vocab[context]})")
# %%
