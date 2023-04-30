

import struct
import tensorflow as tf
import snappy

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
  for ex in unsnappy('mega-v2-9.snappy'):
    n += 1
    if n % mod == 0:
      print(n)
      mod *= 2
  print('done', n)

if __name__ == '__main__':
  app.run(main)
