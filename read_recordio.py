import tensorflow as tf
import time
import functools
import sys, os

from absl import app

FEATURES = {
  'board': tf.io.FixedLenFeature([1280], tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64)
}

def _parse_function(example_proto):
  print('ex: ', example_proto, type(example_proto))
  return tf.io.parse_single_example(example_proto, {
       'board': tf.io.FixedLenFeature([], tf.float32),
       'label': tf.io.FixedLenFeature([], tf.int64)
    })




def main(argv):
  # print('# v1')
  # ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'])
  # for ent in ds.take(1):
  #   print(ent)

  if False: # well, doesn't crash
    print('# v2')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    #ds = ds.map(_parse_function)
    for batch in iter(ds.take(1)):
      print(repr(batch))
      print(type(batch))

  if False:  # good
    print('# v3')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    #ds = ds.map(_parse_function)
    for batch in iter(ds.take(1)):
      print(tf.train.Example().FromString(batch.numpy()))

  if False: # fail
    print('# v4')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    ds = ds.take(1)
    ds = ds.map(_parse_function)
    for ent in iter(ds):
      print('ent', ent)

  if False: # ok
    print('# v44')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    for ent in iter(ds):
      print('ent', type(ent))
      print('ent', ent.dtype)
      print('ent', ent.shape)
      print(ent.numpy()[0:80])
      break

  if False:  # YES
    print('# v44')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    ds = ds.batch(4)
    for ent in iter(ds):
      print('ent', type(ent))
      print('ent', ent.dtype)
      print('ent', ent.shape)
      print('yes: ', tf.io.parse_example(ent, features=FEATURES))
      break

  if True: # YES
    print('# v444')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    ds = ds.batch(4)
    ds = ds.map(functools.partial(tf.io.parse_example, features=FEATURES))
    for ent in iter(ds):
      #print('ent', type(ent))
      #print('ent', ent.dtype)
      #print('ent', ent.shape)
      print(ent)
      #print('yes: ', tf.io.parse_example(ent, features=FEATURES))
      break

  if False: # ?
    print('# v5')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    ds = ds.batch(2)
    ds = ds.take(1)
    for ent in iter(ds):
      print('ent', ent, ent.numpy().shape, ent.shape)





if __name__ == '__main__':
  app.run(main)
