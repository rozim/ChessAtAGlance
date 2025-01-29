# Use sqlitedict as a backing store to shuffle a presumably large
# number of lines from stdin.

import sys
import os
import zlib
import sqlitedict
import sqlite3
from absl import app, flags
import random
import tempfile


def my_encode(obj):
  return zlib.compress(obj.encode('utf-8'))


def my_decode(obj):
  return zlib.decompress(obj).decode('utf-8')


def main(argv):
  t = tempfile.mkstemp(suffix='.db')[1]

  db = sqlitedict.open(t,
                       flag='c',
                       timeout=60,
                       encode=my_encode,
                       decode=my_decode)
  for line in sys.stdin:
    db[str(random.random())] = line.strip()
  keys = list(db.keys())
  random.shuffle(keys)
  for k in keys:
    print(db[k])
  db.close()
  os.remove(t)


if __name__ == '__main__':
  app.run(main)
