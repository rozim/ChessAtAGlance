import tensorflow as tf
import random
import numpy as np
# import pandas as pd
from absl import app
from absl import flags
from absl import logging
#from open_spiel.python.observation import make_observation
import pyspiel
import functools
import time
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten, Add, Conv2D, Permute, Multiply
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Activation, GlobalAveragePooling2D

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Ftrl
from tensorflow.python.keras import backend

from data import BOARD_SHAPE
import warnings
import sys, os

import hashlib



def make_model(num_filters=4):
  data_format = 'channels_last'
  my_conv2d = functools.partial(
    Conv2D,
    filters=num_filters,
    kernel_size=(3, 3),
    data_format=data_format,
    padding='same',
    use_bias=False)

  board = Input(shape=BOARD_SHAPE, name='board', dtype='float32')

  x = board
  x = Permute([2, 3, 1])(x)
  x = BatchNormalization()(x)
  for _ in range(4):
    x = my_conv2d()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = my_conv2d()(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  x = BatchNormalization()(x)
  x = Dense(32, activation='relu')(x)
  x = BatchNormalization()(x)
  d0_norm = Dense(1, activation='tanh', name='d0_norm')(x)
  return Model(inputs=[board], outputs=[d0_norm])


def fixd(d):
  return np.clip(d, -9999.0, 9999.0) / 9999.0


def sha256(foo):
  m = hashlib.sha256()
  m.update(str(foo).encode('utf-8'))
  return m.hexdigest()

def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  game = pyspiel.load_game('chess')

  boards = []
  d0s = []
  nbad = 0
  boards_pct = []
  with open('sf-regression.csv') as f:
    mod = 1
    for row, line in enumerate(f):
      if row == 0:
        continue

      ar = line.strip().split(',')
      fen, d0 = ar[0], fixd(int(ar[1]))
      state = game.new_initial_state(fen)
      if state.current_player() < 0:
        #print('bad: ', fen)
        nbad += 1
        continue
      #print(len(state.observation_tensor()))
      #print(type(state.observation_tensor()))
      try:
        board = np.array(state.observation_tensor(), dtype='float32').reshape((20, 8, 8))
      except pyspiel.SpielError:
        print('bug: ', fen, '::', d0)
        print(state)
        print('p', state.current_player())
        #help(state)
        sys.exit(0)
      boards.append(board)
      if random.random() < 0.01:
        boards_pct.append(board)
      d0s.append(d0)
      if row % mod == 0:
        mod *= 2
        print('row: ', row, 'bad: ', nbad)
      if False and row > 500000:
        break

  print('bad fen: ', nbad)

  seed = int(time.time())
  rng = random.Random(seed)
  rng.shuffle(boards)
  rng = random.Random(seed)
  rng.shuffle(d0s)

  xx = np.stack(boards)
  xx_pct= np.stack(boards_pct)
  yy = np.stack(d0s)

  print('yy', yy)

  #print('shapes: ', boards[0].shape, xx.shape, yy.shape)

  m = make_model()
  #m.summary()

  m.compile(optimizer=Adam(learning_rate=0.10),
            loss={'d0_norm': 'mse'},
            metrics=['mse', 'mae'])

  model = m
  print('IN')
  [print(i.shape, i.dtype) for i in model.inputs]
  print('OUT')
  [print(o.shape, o.dtype) for o in model.outputs]
  print('LAY')
  [print(l.name, l.input_shape, l.dtype) for l in model.layers]
  print('OK')

  #xx = np.random.random((1024, 20, 8, 8)).astype('float32')
  #yy = np.random.random((1024,)).astype('float32')
  #print(xx.shape)
  #print(xx[0].shape)

  #print(xx[0].flatten())
  #print('#')
  #print(xx[1].flatten())
  #sys.exit(0)
  #print(m.predict(xx_pct))
  #m.fit({'board': xx}, {'d0_norm': yy}, batch_size=32, epochs=1)

  p = m.predict(xx_pct)
  pos, neg = 0, 0
  for ent in p:
    ent = ent[0]
    if ent > 0.0:
      pos += 1
    else:
      neg += 1
  print('prelim: ', pos, neg, len(xx_pct))

  m.fit(xx, yy, batch_size=1024, epochs=100)
  p = m.predict(xx_pct)
  print('len', len(p))
  #print('all, p)
  #print('preds: ', m.predict(xx_pct))
  pos, neg, z = 0, 0, 0
  for ent in p:
    ent = ent[0]
    #print('ent: ', type(ent), ent, pos, neg, ent>0.9)
    if ent > 0.0:
      pos += 1
    elif neg < 0.0:
      neg += 1
    else:
      z += 1
  print('dist: ', pos, neg, z, len(p))

if __name__ == '__main__':
  app.run(main)
