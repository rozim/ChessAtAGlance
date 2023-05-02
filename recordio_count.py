
from absl import app

import tensorflow as tf

def main(argv):
  ds = tf.data.TFRecordDataset('data/f1000.rio-00031-of-00100', 'ZLIB')
  n = 0
  mod = 1
  for ent in ds:
    n += 1
    if n % mod == 0:
      print(n)
      mod *= 2
  print('done: ',n)


if __name__ == "__main__":
  app.run(main)
