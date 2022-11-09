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
import tensorflow.keras
import tensorflow.keras.layers

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LayerNormalization, Dense
from tensorflow.keras.layers import Flatten

from keras_nlp_hack import TokenAndPositionEmbedding, TransformerEncoder


def create_transformer_model(num_heads=4,
                             key_dim=32,
                             intermediate_dim=64,
                             num_layers=2,
                             activation='relu'):

  ln = functools.partial(LayerNormalization, epsilon=1e-5)
  inp = Input(name='board', shape=(TRANSFORMER_SIZE,), dtype=tf.int32)

  pos = TokenAndPositionEmbedding(
    vocabulary_size=TRANSFORMER_VOCABULARY,
    sequence_length=TRANSFORMER_SIZE,
    embedding_dim=key_dim)

  y = pos(inp)

  for i in range(num_layers):
    te = TransformerEncoder(intermediate_dim=intermediate_dim, num_heads=num_heads, name=f'transformer_{i}')
    y = te(y)

  y = Flatten()(y)
  y = Dense(TRANSFORMER_VOCABULARY, use_bias=False, activation=None, name='logits')(y)
  # y = ln()(y)

  return Model(inp, y)


def main(argv):
  fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
  board = chess.Board(fen)

  print()
  m = create_transformer_model(num_layers=4)
  m.summary(expand_nested=True)

  enc = encode_transformer_board(board)
  enc = np.reshape(enc, (1,TRANSFORMER_SIZE))

  print('enc=', enc.shape)
  res = m( [{'board': enc}])
  print(res)

if __name__ == "__main__":
  app.run(main)
