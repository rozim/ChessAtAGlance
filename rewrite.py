import sys
import tensorflow as tf
import functools
from encode import CNN_FEATURES

opts = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    output_buffer_size=(4 * 1024 * 1024))

ds = tf.data.TFRecordDataset(['foo.rio'], 'ZLIB')


with tf.io.TFRecordWriter('foo99.rio', opts) as rio:
  for ent in ds:
    rio.write(tf.io.serialize_tensor(ent).numpy())
del ds


print('read1')
ds2 = tf.data.TFRecordDataset(['foo99.rio'], 'ZLIB')
if True:
  ds2 = ds2.batch(1)
  ds2 = ds2.map(functools.partial(tf.io.parse_example, features=CNN_FEATURES))
  for ent in ds2.take(1):
    print(ent)
    break
