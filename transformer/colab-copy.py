import sys
import os
import functools

import numpy as np

import chess
from chess import WHITE, BLACK

from absl import app
from absl import flags

from encode import TRANSFORMER_SIZE
from encode import TRANSFORMER_VOCABULARY
from encode import encode_transformer_board

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.layers import *
import numpy as np
import sys, os, pprint

from keras_nlp_hack import TokenAndPositionEmbedding, TransformerEncoder



te = TransformerEncoder(intermediate_dim=64, num_heads=4)
inp = Input(name='board', shape=(64,), dtype=tf.int32)

pos = TokenAndPositionEmbedding(
        vocabulary_size=10,
        sequence_length=64,
        embedding_dim=32)

y = pos(inp)
for i in range(2):
  te = TransformerEncoder(intermediate_dim=64, num_heads=2)
  y = te(y)

m = Model(inp, y)
m.summary()

data = np.random.randint(10, size=64)
data = np.reshape(data, (1,64))
data.shape

m(data)
