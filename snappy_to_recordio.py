import struct
import tensorflow as tf
import snappy
import time

from absl import app


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
  n = 0
  mod = 1
  t1 = time.time()
  with tf.io.TFRecordWriter('mega-v3-0.recordio', 'ZLIB') as rio:
    for ex in unsnappy('mega-v3-0.snappy'):
      rio.write(ex.SerializeToString())
      n += 1
      if n % mod == 0:
        print(n, int(time.time() - t1))
        mod *= 2
  print()
  print('done', n, int(time.time() - t1))

if __name__ == '__main__':
  app.run(main)          
