
# After generate_transformer_training_data.py runs and stores
# the data in sqlite, read from the db and write out to a compressed
# jsonlines file.
#
# Experiment.

import sqlitedict
import zlib
import gzip
import json
import time

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('db',
                    '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400.db',
                    'DB to read from')

flags.DEFINE_string('out',
                    '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400.jsonl.gz',
                    'Output - compressed json lines')


#def my_json_encode(obj):
#  return zlib.compress(json.dumps(obj).encode('utf-8'))
#def my_json_decode(obj):
#  return json.loads(zlib.decompress(bytes(obj)))


def my_zlib_decode(obj):
  """Return formatted json."""
  return zlib.decompress(bytes(obj))

def main(argv):
  dbio = sqlitedict.open(FLAGS.db,
                         flag='c',
                         timeout=60,
                         decode=my_zlib_decode)

  #with open(FLAGS.out, 'w') as f:
  mod = 1000
  t1 = time.time()
  with gzip.GzipFile(FLAGS.out, 'wb') as f:
    for i, v in enumerate(dbio.values()):
      #f.write(v.decode('utf-8') + '\n')
      f.write(v)
      f.write(b'\n')
      if i % mod == 0:
        t2 = time.time()
        print(i, int(t2 - t1))
        t1 = t2
        mod *= 2
        mod = min(100_000, mod)


  print('written')
  #with open(FLAGS.out, 'r') as f:
  # with gzip.GzipFile(FLAGS.out, 'rb') as f:
  #   for obj in (json.loads(line) for line in f.readlines()):
  #     print('read: ', type(obj), obj)

if __name__ == '__main__':
  app.run(main)
