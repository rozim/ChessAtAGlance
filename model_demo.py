import sys
import tensorflow as tf
import leveldb
from absl import app
from absl import flags
from absl import logging

import glob
import yaml
import re
from contextlib import redirect_stdout
import collections
import datetime
import functools
import itertools
import math
import numpy as np
import os

import random
import sys
import tensorflow as tf
import time

from zipfile import ZipFile
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, LambdaCallback, Callback
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten, Add
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate, Activation
from tensorflow.keras.layers import GaussianNoise, LeakyReLU, Softmax
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, Discretization
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Ftrl
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.python.keras import backend
import pandas as pd


NUM_CLASSES = 4672
BOARD_SHAPE = (20, 8, 8)
BOARD_FLOATS = 1280

def df_to_csv(df, fn, float_format='%6.4f'):
  df.to_csv(fn, index=False, float_format=float_format)

  
class objdict(dict):
  def __getattr__(self, name):
    assert name in self, (name, self.keys())
    return self[name]  


def gen(fn):
  db = leveldb.LevelDB(fn)
  for ent in db.RangeIter():
    ex = tf.train.Example().FromString(ent[1])
    board = tf.convert_to_tensor(ex.features.feature['board'].float_list.value,
                                 dtype=tf.float32)
    board = tf.reshape(board, BOARD_SHAPE)
    action = tf.convert_to_tensor(ex.features.feature['label'].int64_list.value[0],
                                 dtype=tf.int64)
    yield (board, action)


class LogLrCallback(Callback):
  def on_epoch_end(self, epoch, logs):
    logs['lr'] = backend.get_value(self.model.optimizer.lr(epoch))


def main(_argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  gen1 = functools.partial(gen, 'mega-v2-1.leveldb')
  gen2 = functools.partial(gen, 'mega-v2-2.leveldb')
  ds1 = tf.data.Dataset.from_generator(gen1,
                                      output_types=('float32', 'int64'),
                                      output_shapes=(BOARD_SHAPE, []))
  ds1 = ds1.repeat()
  ds1 = ds1.batch(128)

  ds2 = tf.data.Dataset.from_generator(gen2,
                                      output_types=('float32', 'int64'),
                                      output_shapes=(BOARD_SHAPE, []))
  ds2 = ds2.repeat()
  ds2 = ds2.batch(128)

  kernel_regularizer = regularizers.l2(1e-5)
  board = Input(shape=BOARD_SHAPE, dtype='float32')
  x = board
  x = Flatten()(x)
  x = Dense(4096, activation='relu')(x)
  x = Dense(NUM_CLASSES, name='logits', activation=None)(x)
  m = Model(inputs=[board], outputs=x)
  m.summary()
  with open('last-model.txt', 'w') as f:
    with redirect_stdout(f):
      m.summary()

  lr = CosineDecayRestarts(initial_learning_rate=0.001,
                           first_decay_steps=5,
                           t_mul=1,
                           m_mul=1,
                           alpha=0.10)
  m.compile(optimizer=Adam(learning_rate=lr),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
  callbacks = [TerminateOnNaN(),
               LogLrCallback()]
  history = m.fit(x=ds1,
                  epochs=25,
                  steps_per_epoch=256,
                  validation_data=ds2,
                  validation_steps=64,
                  callbacks=callbacks)
  print('all done')
  df = pd.DataFrame(history.history)
  df_to_csv(df, 'last.csv')



  # print('predict: ', m.predict(ds.take(1)))


if __name__ == '__main__':
  app.run(main)
