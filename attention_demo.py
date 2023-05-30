import sys


from absl import app
from absl import flags

import tensorflow as tf

from encode import *


def main(argv):
  fen = 'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1'
  emb_layer = tf.keras.layers.Embedding(input_dim=TRANSFORMER_VOCABULARY, # vocab
                                  output_dim=4,
                                  input_length=TRANSFORMER_LENGTH)

  pos_emb_layer = tf.keras.layers.Embedding(input_dim=TRANSFORMER_LENGTH,
                                            output_dim=4,
                                            input_length=TRANSFORMER_LENGTH)
  pos_indices = tf.range(TRANSFORMER_LENGTH)
  pos_values = pos_emb_layer(pos_indices)
  print("POS: ", pos_values)

  before = chess.Board(fen)
  encoded = encode_transformer_board_wtm(before)
  enc_values = emb_layer(encoded)
  print("ENC: ", enc_values)

  added = tf.keras.layers.Add()([pos_values, enc_values])
  print("ADDED: ", added)


if __name__ == "__main__":
  app.run(main)
