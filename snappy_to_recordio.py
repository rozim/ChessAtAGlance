import struct
import tensorflow as tf
import snappy
import time

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('fn_in', None, '')
flags.DEFINE_string('fn_out', None, '')

def unsnappy(fn):
  with open(fn, 'rb', 8 * 1024 * 1024) as f:
    unpack = struct.unpack
    uncompress  = snappy.uncompress    
    read = f.read    
    while True:
      blob = read(4)
      if len(blob) == 0:
        return
      n = unpack('@i', blob)[0]
      yield tf.train.Example().FromString(snappy.uncompress(read(n)))


def main(argv):
  flags.mark_flags_as_required(['fn_in', 'fn_out'])
  assert FLAGS.fn_in != FLAGS.fn_out
  n = 0
  mod = 16 * 1024
  t1 = time.time()

  opts = tf.io.TFRecordOptions(
    compression_type='ZLIB', 
    output_buffer_size=(4 * 1024 * 1024))

  with tf.io.TFRecordWriter(FLAGS.fn_out, opts) as rio:
    for ex in unsnappy(FLAGS.fn_in):
      rio.write(ex.SerializeToString())
      n += 1
      if n % mod == 0:
        print(n, int(time.time() - t1))
        mod *= 2
  print()
  print('done', n, int(time.time() - t1))

if __name__ == '__main__':
  app.run(main)          
