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
from tensorflow.keras.layers import MultiHeadAttention

def create_transformer_model(num_heads=4,
                             key_dim=32,
                             num_layers=4):
  board = Input(name='board', shape=(TRANSFORMER_SIZE,))
  inputs=[board]

  e1 = Embedding(name='board_embedding',
                 input_dim=TRANSFORMER_VOCABULARY,
                 output_dim=key_dim)(board)

  if True:
    # position_ids = tf.range(TRANSFORMER_SIZE, dtype=tf.int32)[tf.newaxis, :]
    position_ids = Input(name='range', shape=(TRANSFORMER_SIZE,))
    inputs.append(position_ids)

    e2 = Embedding(name='position_embedding',
                   input_dim=TRANSFORMER_SIZE,
                   output_dim=key_dim)(position_ids)

    attn = tf.add(e1, e2, name='pos_add')
  else:
    attn = e1


  for i in range(num_layers):
    attn = tf.keras.layers.MultiHeadAttention(
      num_heads=num_heads,
      key_dim=key_dim)(attn, attn, attn)

  return Model(inputs=inputs, outputs=[attn])


def main(argv):
  fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
  board = chess.Board(fen)

  print()
  m = create_transformer_model()
  m.summary(expand_nested=True)
  print()
  print('layes:')
  for lay in m.layers:
    print(lay)
  enc = encode_transformer_board(board)
  rng = tf.range(TRANSFORMER_SIZE, dtype=tf.int32)[tf.newaxis, :]
  res = m( [{'board': enc, 'range': rng}])
  print(res)

if __name__ == "__main__":
  app.run(main)
