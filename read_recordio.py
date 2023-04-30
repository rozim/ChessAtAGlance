import tensorflow as tf
import time
import functools
import sys, os

from absl import app

from encode import CNN_FLAT_SHAPE

FEATURES = {
  'board': tf.io.FixedLenFeature([CNN_FLAT_SHAPE], tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_function(example_proto):
  print('ex: ', example_proto, type(example_proto))
  return tf.io.parse_single_example(example_proto, {
       'board': tf.io.FixedLenFeature([], tf.float32),
       'label': tf.io.FixedLenFeature([], tf.int64)
    })



def main(argv):
  print('flat: ', CNN_FLAT_SHAPE)
  ds = tf.data.TFRecordDataset(['foo.rio'], 'ZLIB')
  ds = ds.batch(1)
  ds = ds.map(functools.partial(tf.io.parse_example, features=FEATURES))
  for ent in iter(ds):
    print(ent['board'])
    print(ent['label'])
    break



if __name__ == '__main__':
  app.run(main)
