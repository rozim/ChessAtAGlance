
# Best effort to Shuffle the json lines file
# assuming it doesn't fit into memory.

import sqlitedict
import zlib
import gzip
import time
import secrets

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS

flags.DEFINE_string('input',
                    '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400.jsonl.gz',
                    'Input - compressed json lines')
flags.DEFINE_string('output',
                    '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled.jsonl.gz',
                    'Output - compressed json lines')

flags.DEFINE_integer('n',
                 1_000_000,
                 'Size of buffer')


def main(argv):
  t1 = time.time()
  mod, n_read, n_write = 100_000, 0, 0
  buffer = [None] * FLAGS.n
  t1 = time.time()
  f_in = gzip.GzipFile(FLAGS.input, 'rb')
  f_out = gzip.GzipFile(FLAGS.output, 'xb')
  for line in f_in.readlines():
    n_read += 1
    i = secrets.randbelow(FLAGS.n)
    old = buffer[i]
    if old:
      n_write += 1
      f_out.write(old)
    buffer[i] = line
    if n_read % mod == 0:
      print(n_read, n_write, int(time.time() - t1))


  for line in buffer:
    if line:
      f_out.write(line)
  f_out.close()
  f_in.close()


if __name__ == '__main__':
  app.run(main)
