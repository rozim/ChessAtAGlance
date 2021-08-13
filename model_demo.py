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
from tensorflow.keras.losses import BinaryCrossentropy as Loss_BCE
from tensorflow.keras.metrics import AUC
from tensorflow.keras.metrics import BinaryCrossentropy as Metric_BCE
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Ftrl
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.python.keras import backend


NUM_CLASSES = 4672

def gen(fn):
  db = leveldb.LevelDB(fn)
  for ent in db.RangeIter():
    ex = tf.train.Example().FromString(ent[1])
    board = tf.convert_to_tensor(ex.features.feature['board'].float_list.value,
                                 dtype=tf.float32)
    action = tf.convert_to_tensor(ex.features.feature['label'].int64_list.value[0],
                                 dtype=tf.int64)
    yield (board, action)

def main(_argv):
  gen1 = functools.partial(gen, 'mega-v2-1.leveldb')
  gen2 = functools.partial(gen, 'mega-v2-2.leveldb')  
  ds1 = tf.data.Dataset.from_generator(gen1,
                                      output_types=('float32', 'int64'),
                                      output_shapes=([1280,], []))
  ds1 = ds1.batch(128)

  ds2 = tf.data.Dataset.from_generator(gen2,
                                      output_types=('float32', 'int64'),
                                      output_shapes=([1280,], []))
  ds2 = ds2.batch(128)  

  board = Input(shape=(1280,), dtype='float32')  
  x = board
  x = Dense(1024, activation='relu')(x)
  x = Dense(NUM_CLASSES, name='logits', activation=None)(x)
  x = Softmax()(x)
  m = Model(inputs=[board], outputs=x)
  m.summary()

  m.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=[
    #'categorical_accuracy',
                                                                                           'accuracy'])
  m.fit(x=ds1,
        epochs=10,
        steps_per_epoch=64,
        validation_data=ds2,
        validation_steps=16)
  print('all done')
  # print('predict: ', m.predict(ds.take(1)))


if __name__ == '__main__':
  app.run(main)  
