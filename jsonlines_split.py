
# Split a jsonlines gz file into training and test data.
# Assume it is already shuffled.

import gzip
import io

from absl import app
from absl import flags
from absl import logging

import time


FLAGS = flags.FLAGS

flags.DEFINE_string('input',
                    '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled.jsonl.gz',
                    'Input - compressed json lines')
flags.DEFINE_string('train',
                    '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled-train.jsonl.gz',
                    'Output - train')
flags.DEFINE_string('test',
                    '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled-test.jsonl.gz',
                    'Output - test')

flags.DEFINE_integer('pc', 90, 'Percent train')


def main(argv):
  t1 = time.time()
  f_in = gzip.GzipFile(FLAGS.input, 'rb')

  # >= 100MB with compresslevel=1 seems very fast
  buffer_size = 250 * 1024 * 1024
  new_code = True

  if new_code:
    f1 = open(FLAGS.train, 'xb')
    f2 = open(FLAGS.test, 'xb')
    gz1 = gzip.GzipFile(fileobj=f1, mode='wb', compresslevel=1)
    gz2 = gzip.GzipFile(fileobj=f2, mode='wb', compresslevel=1)
    io1 = io.BufferedWriter(gz1, buffer_size=buffer_size)
    io2 = io.BufferedWriter(gz2, buffer_size=buffer_size)
  else:
    f_train = gzip.GzipFile(FLAGS.train, 'xb')
    f_test = gzip.GzipFile(FLAGS.test, 'xb')

  for i, line in enumerate(f_in):
    if i % 10_0000 == 0:
      dt = time.time() - t1
      print(f'{i:10,} {int(i/dt):10,} {dt:.0f}s')

    if new_code:
      if i % 100 < FLAGS.pc:
        io1.write(line)
      else:
        io2.write(line)
    else:
      if i % 100 < FLAGS.pc:
        f_train.write(line)
      else:
        f_test.write(line)

  if new_code:
    io1.close()
    io2.close()
    gz1.close()
    gz2.close()
    f1.close()
    f2.close()
  else:
    f_out.close()
    f_train.close()
    f_test.close()


if __name__ == '__main__':
  app.run(main)
