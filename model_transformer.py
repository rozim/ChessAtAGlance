import sys
import os
import functools

import numpy as np
import pandas as pd

import chess
from chess import WHITE, BLACK

from absl import app
from absl import flags


import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.layers

from tensorflow.keras import Model, Input
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.optimizers.experimental import AdamW
# from tensorflow.keras.optimizers.legacy import AdamW
from tensorflow.keras.optimizers import Adam

from encode import *
from pychess_util import *
from plan import load_plan

# https://www.tensorflow.org/api_docs/python/tfm/nlp/layers/PositionEmbedding
class PositionEmbedding(tf.keras.layers.Layer):
  """Creates a positional embedding.

  Example:
  ```python
  position_embedding = PositionEmbedding(max_length=100)
  inputs = tf.keras.Input((100, 32), dtype=tf.float32)
  outputs = position_embedding(inputs)
  ```


  Args:
    max_length: The maximum size of the dynamic sequence.
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    seq_axis: The axis of the input tensor where we add the embeddings.

  Reference: This layer creates a positional embedding as described in
  [BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding](https://arxiv.org/abs/1810.04805).
  """

  def __init__(self,
               max_length,
               initializer="glorot_uniform",
               seq_axis=1,
               **kwargs):

    super().__init__(**kwargs)
    if max_length is None:
      raise ValueError(
          "`max_length` must be an Integer, not `None`."
      )
    self._max_length = max_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._seq_axis = seq_axis

  def get_config(self):
    config = {
        "max_length": self._max_length,
        "initializer": tf.keras.initializers.serialize(self._initializer),
        "seq_axis": self._seq_axis,
    }
    base_config = super(PositionEmbedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    dimension_list = input_shape.as_list()
    width = dimension_list[-1]
    weight_sequence_length = self._max_length

    self._position_embeddings = self.add_weight(
        "embeddings",
        shape=[weight_sequence_length, width],
        initializer=self._initializer)

    super().build(input_shape)

  def call(self, inputs):
    input_shape = tf.shape(inputs)
    actual_seq_len = input_shape[self._seq_axis]
    position_embeddings = self._position_embeddings[:actual_seq_len, :]
    new_shape = [1 for _ in inputs.get_shape().as_list()]
    new_shape[self._seq_axis] = actual_seq_len
    new_shape[-1] = position_embeddings.get_shape().as_list()[-1]
    position_embeddings = tf.reshape(position_embeddings, new_shape)
    return tf.broadcast_to(position_embeddings, input_shape)


# # https://stackoverflow.com/questions/55235212/model-summary-cant-print-output-shape-while-using-subclass-model
# class PositionEmbeddingLayer(tf.keras.layers.Layer):
#   def __init__(self, sequence_length, vocab_size, output_dim, name="position_embedding", **kwargs):
#     super(PositionEmbeddingLayer, self).__init__(name=name, **kwargs)
#     self.sequence_length = sequence_length
#     self.vocab_size = vocab_size
#     self.output_dim = output_dim


#   def build(self, input_shape):
#     print("BUILD: ", input_shape)
#     self.word_embedding_layer = tf.keras.layers.Embedding(
#       input_dim=self.vocab_size,
#       output_dim=self.output_dim,
#       input_length=input_shape,
#       name="token_table",
#     )
#     self.position_embedding_layer = tf.keras.layers.Embedding(
#       input_dim=self.sequence_length,
#       output_dim=self.output_dim,
#       input_length=input_shape,
#       name="position_table",
#     )

#     self.add_layer = tf.keras.layers.Add()

#   def call(self, inputs):
#     position_indices = tf.range(tf.shape(inputs)[-1])
#     embedded_words = self.word_embedding_layer(inputs)
#     embedded_indices = self.position_embedding_layer(position_indices)
#     return self.add_layer([embedded_words, embedded_indices])

#   @property
#   def layers(self):
#     return [
#       self.word_embedding_layer,
#       self.position_embedding_layer,
#       self.add_layer]


def create_transformer_model(mplan):
  if hasattr(mplan, 'l2'):
    kernel_regularizer = regularizers.l2(mplan.get('l2', 0.0))
  else:
    kernel_regularizer = None
  kernel_initializer = mplan.get('kernel', 'random_uniform')

  embedding_dim = mplan.embedding_dim
  intermediate_dim = mplan.intermediate_dim

  my_ln = LayerNormalization
  my_act = functools.partial(Activation, activation=mplan.get('activation', 'relu'))
  my_dense = functools.partial(Dense,
                               kernel_regularizer=kernel_regularizer,
                               kernel_initializer=kernel_initializer)

  emb_layer = tf.keras.layers.Embedding(input_dim=TRANSFORMER_VOCABULARY,
                                        output_dim=embedding_dim,
                                        input_length=TRANSFORMER_LENGTH,
                                        name="token_table")

  pos_layer = PositionEmbedding(max_length=TRANSFORMER_LENGTH)

  board = Input(shape=TRANSFORMER_LENGTH, name='board', dtype='int32')

  x = board
  x = emb_layer(x)
  x = Add()([pos_layer(x), x])
  x = my_dense(intermediate_dim, name='project')(x) # project so skip works
  x = my_ln()(x)

  for lnum in range(mplan.num_layers):
    x = MultiHeadAttention(num_heads=mplan.num_heads,
                           key_dim=intermediate_dim,
                           value_dim=intermediate_dim,
                           kernel_regularizer=kernel_regularizer,
                           kernel_initializer=kernel_initializer,
                           dropout=mplan.dropout,
                           name=f'mha_{lnum}')(query=x, key=x, value=x)

  # for lnum in range(0):
  #   skip = x
  #   x = tf.keras.layers.Attention(name=f'attn_{lnum}a')([x, x])
  #   x = my_ln(name=f'ln_{lnum}a')(x) # check order
  #   x = Add(name=f'add_{lnum}a')([x, skip])

  #   skip = x
  #   x = Dense(intermediate_dim, name=f'dense_{lnum}b')(x)
  #   x = my_ln(name=f'ln_{lnum}b')(x)
  #   x = Activation(name=f'act_{lnum}b', activation='gelu')(x)
  #   x = Add(name=f'add_{lnum}b')([x, skip])

  x = Flatten()(x)
  x = my_ln()(x)
  y = my_dense(NUM_CLASSES, name='logits', activation=None, use_bias=False)(x)
  return Model(inputs=[board], outputs=[y])


def gen():
  for game in gen_games('t2.pgn'):
    for (move, board) in gen_moves(game):
      enc_board, enc_move = encode_transformer_board_move_wtm(board, move)
      yield enc_board, enc_move

def main(argv):
  plan = load_plan('config/transformer_1.toml')
  model = create_transformer_model(mplan=plan.model)
  model.summary(expand_nested=True)
  for t in model.inputs:
    print("INPUT: ", t)
  for t in model.outputs:
    print("OUTPUT: ", t)

  for v in model.trainable_variables:
    print("VAR: ", v.name, v.shape, v.dtype)

  ds = tf.data.Dataset.from_generator(gen,
                                      output_signature=(
                                        tf.TensorSpec(shape=(69,), dtype=tf.int32),
                                        tf.TensorSpec(shape=(), dtype=tf.int32)))
  ds = ds.batch(1)

  for enc_board, enc_move in ds:
    print('ENT', enc_board.numpy(), ' :: ', enc_move.numpy())
    infer = model(enc_board)
    for foo in infer:
      print('INFER', foo.shape, foo.dtype)
    break


if __name__ == "__main__":
  app.run(main)
