import sys
import os
import functools

import numpy as np
import pandas as pd

import chess
from chess import WHITE, BLACK

from absl import app
from absl import flags

from encode import CNN_FEATURES, NUM_CLASSES, CNN_SHAPE_3D

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

# class Linear(tensorflow.keras.layers.Layer):
#   def __init__(self, units=32, input_dim=32):
#     super().__init__()
#     w_init = tf.random_normal_initializer()
#     self.w = tf.Variable(
#       initial_value=w_init(shape=(input_dim, units), dtype="float32"),
#       trainable=False, # bias only
#     )
#     b_init = tf.zeros_initializer()
#     self.b = tf.Variable(
#       initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
#     )

#   def call(self, inputs):
#     return tf.matmul(inputs, self.w) + self.b


def create_simple_model():
  board = Input(shape=CNN_SHAPE_3D, name='board', dtype='float32')
  x = board
  # in: bs, chan, x, y
  #         16    8, 8
  #     0   1     2  3
  x = Permute([2, 3, 1])(x)
  x = Flatten()(x)
  x = Dense(10 * NUM_CLASSES, activation='relu')(x)
  y = Dense(NUM_CLASSES, name='logits', activation=None, use_bias=False)(x)
  return Model(inputs=[board], outputs=y)


def create_model(mplan):
  data_format = 'channels_last'
  kernel_regularizer = regularizers.l2(1e-6)
  my_conv2d = functools.partial(
    Conv2D,
    filters=mplan.num_filters,
    kernel_size=(3, 3),
    kernel_regularizer=kernel_regularizer,
    data_format=data_format,
    padding='same',
    use_bias=False)
  #my_ln = functools.partial(LayerNormalization, epsilon=1e-5)
  my_ln = LayerNormalization
  my_act = functools.partial(Activation, mplan.activation)

  board = Input(shape=CNN_SHAPE_3D, name='board', dtype='float32')
  x = board
  # in: bs, chan, x, y
  #         16    8, 8
  #     0   1     2  3
  x = Permute([2, 3, 1])(x)

  # Get to right size so skip can work
  x = my_conv2d()(x)
  x = my_ln()(x)
  x = my_act()(x)

  for _ in range(mplan.num_layers):
    skip = x

    x = my_conv2d()(x)
    x = my_ln()(x)
    x = my_act()(x)

    x = my_conv2d()(x)
    x = my_ln()(x)
    x = Add()([x, skip])
    x = my_act()(x)

  x = Flatten()(x)
  y = Dense(NUM_CLASSES, name='logits', activation=None, use_bias=False)(x)
  return Model(inputs=[board], outputs=y)


def df_to_csv(df, fn, float_format='%6.4f'):
  df.to_csv(fn, index=False, float_format=float_format)

def extract(blob):
  t = tf.io.parse_example(blob, features=CNN_FEATURES)
  return t['board'], t['label']

def main(argv):
  from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, LambdaCallback, Callback

  model = create_model()
  model.summary()

  ds = tf.data.TFRecordDataset(['foo.rio'], 'ZLIB')
  ds = ds.batch(16)
  ds = ds.map(extract)
  ds = ds.repeat()

  model.compile(optimizer=Adam(learning_rate=0.001),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
  callbacks = [TerminateOnNaN(),
               # LogLrCallback()
               ]
  history = model.fit(x=ds.take(128).repeat(),
                  epochs=25,
                  steps_per_epoch=100,
                  #validation_data=ds2,
                  #validation_steps=64,
                  callbacks=callbacks)

  df = pd.DataFrame(history.history)
  df_to_csv(df, '/dev/stdout')

if __name__ == "__main__":
  app.run(main)
