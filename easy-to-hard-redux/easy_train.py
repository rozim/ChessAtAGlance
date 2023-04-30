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
from tensorflow.keras.losses import CategoricalCrossentropy

#FLAT_SHAPE = (12 * 8 * 8,)
SHAPE_3D = (12, 8, 8)
FEATURES = {
  'puzzle': tf.io.FixedLenFeature(SHAPE_3D, tf.float32),
  'solution': tf.io.FixedLenFeature([8, 8, 1], tf.float32),
  'rating': tf.io.FixedLenFeature(1, tf.int64),
  'refutation_uci': tf.io.FixedLenFeature(1, tf.string),
}

class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]


def _extract(blob):
  t = tf.io.parse_example(blob, features=FEATURES)
  return ((t['puzzle'],),
          t['solution'])


class BasicBlock(layers.Layer):
  def __init__(self, num_filters, l2):
    super(BasicBlock, self).__init__()
    self.num_filters = num_filters
    self.l2 = l2

  def build(self, input_shape):
    kernel_regularizer = None
    data_format = 'channels_last' # (batch_size, height, width, channels)
    my_conv2d = functools.partial(
      Conv2D,
      filters=self.num_filters,
      kernel_size=(3, 3),
      kernel_regularizer=kernel_regularizer,
      data_format=data_format,
      padding='same', # padding=1 in paper
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

  board = Input(shape=SHAPE_3D, name='puzzle', dtype='float32')
  x = board
  #x = Reshape(SHAPE_3D, input_shape=FLAT_SHAPE)(board)
  # in: bs, chan, x, y
  #         20    8, 8
  #     0   1     2  3
  x = Permute([2, 3, 1])(x)
  # out: bs, x, y, chan
  #          8, 8, 20

  # Project to right size so skip connections work.
  x = my_conv2d(name=f'cnn_project')(x)
  x = ReLU()(x)

  block = BasicBlock(mplan.num_filters, mplan.l2)
  for i in range(mplan.num_layers):
    x = block(x)

  x = Conv2D(filters=32,
             kernel_size=(3, 3),
             kernel_regularizer=kernel_regularizer,
             data_format=data_format,
             padding='same',
             use_bias=False,
             name='head_conv2')(x)
  x = Conv2D(filters=8,
             kernel_size=(3, 3),
             kernel_regularizer=kernel_regularizer,
             data_format=data_format,
             padding='same',
             use_bias=False,
             name='head_conv3')(x)
  x = Conv2D(filters=1,
             kernel_size=(3, 3),
             kernel_regularizer=kernel_regularizer,
             data_format=data_format,
             padding='same',
             use_bias=False,
             name='head_conv4')(x)

  return Model(inputs=[board], outputs=x)



def get_data(tplan):
  ds = tf.data.TFRecordDataset(['easy-v3.rio'], 'ZLIB', num_parallel_reads=1)
  ds = ds.map(_extract)
  ds = ds.batch(tplan.batch_size)
  return ds


class LogLrCallback(Callback):
  def on_epoch_end(self, epoch, logs):
    try:
      logs['lr'] = float(backend.get_value(self.model.optimizer.lr(epoch)))
    except TypeError:
      logs['lr'] = float(backend.get_value(self.model.optimizer.lr))


def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  if True:
    ds = tf.data.TFRecordDataset(['easy-v3.rio'], 'ZLIB', num_parallel_reads=1)
    ds = ds.map(_extract)
    ds = ds.batch(2)
    for features, label in ds:
      for f in features:
        print('f: ', f)
      print('label: ', label)

      break

  mplan = toml.load('easy.toml', objdict)
  model = create_model(mplan.model)
  model.summary()

  tplan = mplan.train
  optimizer = tf.keras.optimizers.SGD(
    learning_rate=tplan.lr,
    momentum=tplan.momentum
  )

  model.compile(optimizer=optimizer,
                loss=CategoricalCrossentropy())

  ds = get_data(tplan)

  callbacks = [TerminateOnNaN(),
               LogLrCallback()]

  print('# before fit')


  history = model.fit(x=ds,
                  epochs=tplan.epochs,
                  steps_per_epoch=tplan.steps_per_epoch,
                  callbacks=callbacks)
  print('# after fit')

  ds = get_data(tplan)
  foo = model.evaluate(x=ds, return_dict=True, steps=1)
  print('foo: ', foo)
  res = model.predict(x=ds,
                      batch_size=7,
                      steps=3)
  print('res: ', res.shape, res)


if __name__ == '__main__':
  app.run(main)
