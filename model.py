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

from encode import CNN_FEATURES, NUM_CLASSES, CNN_SHAPE_3D
from plan import load_plan

DATA_FORMAT = 'channels_last'


class BiasOnly(tensorflow.keras.layers.Layer):
  def __init__(self, units, input_dim):
    super().__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
      initial_value=w_init(shape=(input_dim, units), dtype="float32"),
      trainable=False, # bias only
    )
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
      initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
    )

  def call(self, inputs):
    return tf.stop_gradient(tf.matmul(inputs, self.w)) + self.b
    #return tf.matmul(inputs, self.w) + self.b

class Linear(tensorflow.keras.layers.Layer):
  def __init__(self, units=32, input_dim=32):
    super().__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
      initial_value=w_init(shape=(input_dim, units), dtype="float32"),
      trainable=False, # bias only
    )
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
      initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
    )

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


def create_bias_only_model(mplan):
  board = Input(shape=CNN_SHAPE_3D, name='board', dtype='float32')
  x = board
  x = Flatten()(x)
  y = BiasOnly(NUM_CLASSES, 1024)(x)
  return Model(inputs=[board], outputs=y)


def create_simple_model(mplan):
  kernel_regularizer = regularizers.l2(mplan.l2)

  board = Input(shape=CNN_SHAPE_3D, name='board', dtype='float32')
  x = board
  x = Flatten()(x)

  for _ in range(mplan.num_layers):
    # Skip LN for now.
    x = Dense(NUM_CLASSES, activation=mplan.activation, kernel_regularizer=kernel_regularizer)(x)

  y = Dense(NUM_CLASSES, name='logits', activation=None, use_bias=False, kernel_regularizer=kernel_regularizer)(x)
  return Model(inputs=[board], outputs=y)


def create_cnn_model(mplan):
  if hasattr(mplan, 'l2'):
    kernel_regularizer = regularizers.l2(mplan.get('l2', 0.0))
  else:
    kernel_regularizer = None

  kernel_initializer = mplan.get('kernel', 'random_uniform')

  my_conv2d = functools.partial(
    Conv2D,
    filters=mplan.get('num_filters', 1),
    kernel_size=(3, 3),
    kernel_regularizer=kernel_regularizer,
    kernel_initializer=kernel_initializer,
    data_format=DATA_FORMAT,
    padding='same',
    use_bias=False)
  #my_ln = functools.partial(LayerNormalization, epsilon=1e-5)
  my_ln = LayerNormalization
  my_act = functools.partial(Activation, activation=mplan.get('activation', 'relu'))
  my_dense = functools.partial(Dense, kernel_regularizer=kernel_regularizer,
                               kernel_initializer=kernel_initializer)

  board = Input(shape=CNN_SHAPE_3D, name='board', dtype='float32')
  x = board
  # in: bs, chan, x, y
  #         16    8, 8
  #     0   1     2  3
  x = Permute([2, 3, 1])(x)

  # If not set then assume we are skipping everything.
  if hasattr(mplan, 'num_filters'):
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

  # Historical note: flatten_1x1 removed here, was proven worse
  # than leaving it out.

  x = Flatten()(x)

  for i, w in enumerate(mplan.top_tower):
    x = my_dense(w, name=f'top_{i}', activation=None)(x)
    x = my_ln()(x)
    x = my_act()(x)

  y = my_dense(NUM_CLASSES, name='logits', activation=None, use_bias=False)(x)
  return Model(inputs=[board], outputs=y)




def df_to_csv(df, fn, float_format='%6.4f'):
  df.to_csv(fn, index=False, float_format=float_format)

def extract(blob):
  t = tf.io.parse_example(blob, features=CNN_FEATURES)
  return t['board'], t['label']

def main(argv):
  from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, LambdaCallback, Callback

  plan = load_plan('config/cnn_105.toml')
  model = create_cnn_model(mplan=plan.model)
  model.summary(expand_nested=True)

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
