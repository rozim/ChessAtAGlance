import warnings
import sys, os
from open_spiel.python.observation import make_observation
import tensorflow as tf
import numpy as np
import pandas as pd
import functools

import pyspiel
from absl import app
from absl import flags
from absl import logging
import time
import toml

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, LambdaCallback, Callback
from tensorflow.keras.layers import Flatten, Add, Conv2D, Permute, Reshape
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate, Activation
from tensorflow.keras.layers import Softmax, ReLU

from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from tensorflow.python.keras import backend

NUM_CLASSES = 4672

FEATURES = {
  'board': tf.io.FixedLenFeature((1280,), tf.float32),
  'best_move': tf.io.FixedLenFeature([], tf.int64),
  'rating': tf.io.FixedLenFeature([], tf.int64),
  'uci': tf.io.FixedLenFeature([], tf.string),
}

class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]

def _extract(blob):
  t = tf.io.parse_example(blob, features=FEATURES)
  return ((t['board'], t['rating'], t['uci']),
          t['best_move'])


class ResnetBlock(layers.Layer):
  def __init__(self, num_filters, l2):
    super(ResnetBlock, self).__init__()
    self.num_filters = num_filters
    self.l2 = l2

  def build(self, input_shape):
    kernel_regularizer = None
    data_format = 'channels_last'
    my_conv2d = functools.partial(
      Conv2D,
      filters=self.num_filters,
      kernel_size=(3, 3),
      kernel_regularizer=kernel_regularizer,
      data_format=data_format,
      padding='same',
      use_bias=False)

    self.c1 = my_conv2d()
    self.relu1 = ReLU()
    self.add = Add()
    self.c2 = my_conv2d()
    self.relu2 = ReLU()


  def call(self, x):
    skip = x
    x = self.c1(x)
    x = self.relu1(x)
    x = self.c2(x)

    x = self.add([x, skip])

    x = self.relu2(x)
    return x


  def get_config(self):
    return {'num_filters': self.num_filters,
            'l2': self.l2}


def create_model(mplan):
  kernel_regularizer = regularizers.l2(mplan.l2)
  data_format = 'channels_last'

  my_conv2d = functools.partial(
    Conv2D,
    filters=mplan.num_filters,
    kernel_size=(3, 3),
    kernel_regularizer=kernel_regularizer,
    data_format=data_format,
    padding='same',
    use_bias=False)

  my_dense = functools.partial(Dense,
                               kernel_regularizer=kernel_regularizer)

  board = Input(shape=(1280,), name='board', dtype='float32')
  x = Reshape((20, 8, 8), input_shape=(1280,))(board)
  # in: bs, chan, x, y
  #         20    8, 8
  #     0   1     2  3
  x = Permute([2, 3, 1])(x)
  # out: bs, x, y, chan
  #          8, 8, 20

  # Project to right size so skip connections work.
  x = my_conv2d(name=f'cnn_project')(x)
  x = ReLU()(x)

  # for i in range(mplan.num_resnet_blocks):
  #   skip = x
  #   x = my_conv2d(name=f'cnn_{i}a')(x)
  #   x = ReLU(name=f'relu_{i}a')(x)

  #   x = my_conv2d(name=f'cnn_{i}b')(x)
  #   x = Add(name='skip_{}b'.format(i))([x, skip])
  #   x = ReLU(name=f'relu_{i}b')(x)

  blocks = [ResnetBlock(mplan.num_filters, mplan.l2) for _ in range(4)]
  for j in range(2):
    for i in range(mplan.num_resnet_blocks):
      for block in blocks:
        x = block(x)

  x = Flatten()(x)

  for i, w in enumerate(mplan.top_tower):
    x = my_dense(w, name=f'top_{i}')(x)
    x = ReLU()(x)

  x = my_dense(NUM_CLASSES, name='logits', activation=None)(x)

  return Model(inputs=[board], outputs=x)


def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  if True:
    ds = tf.data.TFRecordDataset(['easy.rio'], 'ZLIB', num_parallel_reads=1)
    ds = ds.map(_extract)
    ds = ds.batch(2)
    for ent in ds:
      print('b', ent[0][0])
      print('r', ent[0][1].numpy())
      print('uci', ent[0][2])
      print('action', ent[1].numpy())
      break


  mplan = toml.load('easy.toml', objdict)
  model = create_model(mplan.model)
  model.summary()


if __name__ == '__main__':
  app.run(main)
