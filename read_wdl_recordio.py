import tensorflow as tf
import time
import functools
import sys, os

from absl import app

from encode import TRANSFORMER_WDL_FEATURES_FEN

def main(argv):
  ds = tf.data.TFRecordDataset(['wdl.rio'], 'ZLIB')
  ds = ds.batch(1)
  ds = ds.map(functools.partial(tf.io.parse_example, features=TRANSFORMER_WDL_FEATURES_FEN))
  for ent in iter(ds):
    print('board', ent['board'])
    print('result', ent['result'])
    print('fen', ent['fen'])
    break



if __name__ == '__main__':
  app.run(main)
