from contextlib import redirect_stdout
import os
import os.path
import pickle
import random
import sys
import time
import warnings
import zlib

from absl import app
from absl import flags
from absl import logging

import sqlite3

import tensorflow as tf

flags.DEFINE_string('out', 'data/mega-v2.rio', 'Recordio file')
flags.DEFINE_integer('shards', 100, 'Number of shards')
FLAGS = flags.FLAGS

def my_decode(obj):
  return pickle.loads(zlib.decompress(bytes(obj)))

def main(argv):
  assert FLAGS.out
  assert FLAGS.shards > 0

  last = time.time()

  opts = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    output_buffer_size=(4 * 1024 * 1024))

  if FLAGS.shards == 1:
    rio = [tf.io.TFRecordWriter(FLAGS.out, opts)]
  else:
    rio = [
      tf.io.TFRecordWriter(f'{FLAGS.out}-{shard:05d}-of-{FLAGS.shards:05d}', opts)
      for shard in range(FLAGS.shards)]


  conn = sqlite3.connect('data2/mega.sqlite')

  ids = list(range(100))
  random.shuffle(ids)

  rows = 0
  for id in ids:
    cursor = conn.cursor()
    sql = f"SELECT value FROM unnamed WHERE CAST(ABS(key + rowid) AS INTEGER) % 100 == {id} ORDER BY RANDOM()"
    print("QUERY: ", sql)
    cursor.execute(sql)
    for row in cursor.fetchall():
      pb = my_decode(row[0])
      rio[random.randint(0, FLAGS.shards-1)].write(pb.SerializeToString())
      rows += 1
      now = time.time()
      if now >= (last + 60.0):
        print(rows, int(now - last))
        last = now
    cursor.close()

  print()
  print("ALL DONE")
  for fh in rio:
    fh.close()


  conn.close()



if __name__ == "__main__":
  app.run(main)
