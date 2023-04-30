import struct
import tensorflow as tf
import leveldb
import time
import sys, os

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('fn_in', None, '')
flags.DEFINE_string('fn_out', None, '')

def main(argv):
  flags.mark_flags_as_required(['fn_in', 'fn_out'])
  assert FLAGS.fn_in != FLAGS.fn_out
  n = 0
  mod = 64 * 1024
  t1 = time.time()

  opts = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    output_buffer_size=(4 * 1024 * 1024))

  with tf.io.TFRecordWriter(FLAGS.fn_out, opts) as rio:
    db = leveldb.LevelDB(FLAGS.fn_in)
    for ent in db.RangeIter():
      # Yuck. How to decode bytearray w/o parsing Example.
      rio.write(tf.train.Example().FromString(ent[1]).SerializeToString())
      n += 1
      if n % mod == 0:
        print(n, int(time.time() - t1))
        mod *= 2
  print()
  print('done', n, int(time.time() - t1))

if __name__ == '__main__':
  app.run(main)
