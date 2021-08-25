import tensorflow as tf
import time

from absl import app


def _parse_function(example_proto):
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

  if True:  # good
    print('# v3')
    ds = tf.data.TFRecordDataset(['mega-v3-0.recordio'], 'ZLIB')
    #ds = ds.map(_parse_function)
    for batch in iter(ds.take(1)):
      print(tf.train.Example().FromString(batch.numpy()))




if __name__ == '__main__':
  app.run(main)          
