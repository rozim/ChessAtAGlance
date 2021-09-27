import warnings
import sys, os
from open_spiel.python.observation import make_observation
import tensorflow as tf
import numpy as np
import pandas as pd

import pyspiel
from absl import app
from absl import flags
from absl import logging
import time

FEATURES = {
  'board': tf.io.FixedLenFeature((1280,), tf.float32),
  'best_move': tf.io.FixedLenFeature([], tf.int64),
  'rating': tf.io.FixedLenFeature([], tf.int64),
  'uci': tf.io.FixedLenFeature([], tf.string),
}

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _extract(blob):
  t = tf.io.parse_example(blob, features=FEATURES)
  return ((t['board'], t['rating'], t['uci']),
          t['best_move'])


def main(argv):
  #tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
    sys.exit(0)

  game = pyspiel.load_game('chess')

  df = pd.read_csv('lichess_db_puzzle.csv')
  t0 = time.time()
  mod = 1024
  opts = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    output_buffer_size=(4 * 1024 * 1024))

  #rio.write(ex.SerializeToString())
  df = df.head(100000)
  df = df.sort_values('Rating')
  with tf.io.TFRecordWriter('easy.rio', opts) as rio:

    index = -1
    for db_index, row in df.iterrows():
      index += 1
      rating = row['Rating']
      best_move = row['Moves'].split(' ')[0] # first
      fen = row['FEN']
      state = game.new_initial_state(fen)

      if state.current_player() < 0:
        continue
      action = state.parse_move_to_action(best_move) # int
      board = state.observation_tensor()

      feature = {
        'board': tf.train.Feature(float_list=tf.train.FloatList(value=board)),
        'uci':  _bytes_feature(best_move.encode('utf-8')),
        'rating':  _int64_feature(rating),
        'best_move': _int64_feature(action),
      }
      pb = tf.train.Example(features=tf.train.Features(feature=feature))
      rio.write(pb.SerializeToString())

      if index % mod == 0:
        dt = int(time.time() - t0)
        print(index, dt, int(index/(dt+1)), int((100 * index) / 1857575), '%')
        mod *= 2


if __name__ == '__main__':
  app.run(main)
