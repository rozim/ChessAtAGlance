

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
      n = unpack('@i', read(4))[0]
      yield tf.train.Example().FromString(snappy.uncompress(read(n)))



def main(argv):
  for ex in unsnappy('mega-v2-9.snappy'):
    print(ex)
    break

if __name__ == '__main__':
  app.run(main)          
