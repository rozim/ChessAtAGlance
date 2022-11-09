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

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Flatten, Reshape, Input, Dense, Reshape, Activation
from tensorflow.keras.layers import Permute, Conv2D, LayerNormalization, Conv2DTranspose
from tensorflow.keras.layers import Add, Multiply, Embedding
# from tensorflow.keras.layers import MultiHeadAttention

from keras_nlp_hack import PositionEmbedding, TransformerEncoder


# class AddPositionEmbedding(tf.keras.layers.Layer):
#   def __init__(self, seq_len, vocab_size, dim, **kwargs):
#     super().__init__(**kwargs)
#     self._seq_len = seq_len
#     self._e = Embedding(vocab_size, dim)

#   def call(self, inputs):
#     return inputs + self._e(np.arange(self._seq_len, dtype='int32'))


def create_transformer_model(num_heads=4,
                              key_dim=32,
                              intermediate_dim=64,
                              num_layers=2):
  pass

# def create_transformer_model(num_heads=4,
#                              key_dim=32,
#                              intermediate_dim=64,
#                              num_layers=2):
#   # self._attention_head_size = int(feature_size // self.num_heads)
#   ln = functools.partial(LayerNormalization, epsilon=1e-05)
#   dense = functools.partial(Dense,
#                             activation='relu',
#                             kernel_initializer='glorot_uniform',
#                             bias_initializer='zeros')

#   board = Input(name='board', shape=(TRANSFORMER_SIZE,))
#   inputs=[board]

#   e_board = Embedding(name='board_embedding',
#                       input_dim=TRANSFORMER_VOCABULARY,
#                       output_dim=key_dim)(board)

#   attn = AddPositionEmbedding(seq_len=TRANSFORMER_SIZE, vocab_size=TRANSFORMER_VOCABULARY, dim=key_dim)(e_board)

#   print('before', attn.shape)
#   attn = MultiHeadAttention(
#     num_heads=num_heads,
#     key_dim=key_dim)(attn, attn, attn)
#   print('after', attn.shape)

#   for i in range(num_layers):
#     print(f'i={i} attn=', attn.shape)
#     orig = attn
#     attn = MultiHeadAttention(
#       num_heads=num_heads,
#       key_dim=key_dim,
#       name=f'mh_{i}')(attn, attn, attn)

#     attn = ln(name=f'ln_{i}a')(attn + orig)

#     orig = attn
#     attn = dense(intermediate_dim, name=f'dense_{i}a')(attn)
#     attn = dense(key_dim, name=f'dense_{i}b')(attn)
#     attn = ln(name=f'ln_{i}b')(attn + orig)

#   return Model(inputs=inputs, outputs=[attn])


def main(argv):
  fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
  board = chess.Board(fen)

  print()
  m = create_transformer_model(num_layers=0)
  m.summary(expand_nested=True)

  # print()
  # print('layes:')
  # for lay in m.layers:
  #   print(lay)

  enc = encode_transformer_board(board)
  print('enc=', enc.shape)
  res = m( [{'board': enc}])
  print(res)

if __name__ == "__main__":
  app.run(main)
