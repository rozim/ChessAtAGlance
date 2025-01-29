
# Split a jsonlines file into training and test data.
# Assume it is already shuffled.
# Simplified version w/o compression from ../jsonlines_split.py

import hashlib
import os
import random
import gzip

import jsonlines
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input',
                    'data/mega2600_shuffled_dedup.jsonl',
                    'Input - uncompressed json lines')

flags.DEFINE_string('train',
                    'data/mega2600_shuffled_dedup_train.json',
                    'Output - train')
flags.DEFINE_string('test',
                    'data/mega2600_shuffled_dedup_test.json',
                    'Output - test')


flags.DEFINE_integer('pc', 99, 'Percent train')


def smart_output(fn, mode='w'):
  if fn.endswith('.gz'):
    return gzip.open(fn, mode)
  return open(fn, mode)


def smart_input(fn, mode='r'):
  if fn.endswith('.gz'):
    return gzip.open(fn, mode)
  return open(fn, mode)


def main(argv):
  assert os.path.exists(FLAGS.input)
  assert not os.path.exists(FLAGS.train)
  assert not os.path.exists(FLAGS.test)
  assert FLAGS.train != FLAGS.test

  nonce = str(random.random()).encode('utf-8')

  with (smart_input(FLAGS.train, 'wt') as f1,
        smart_input(FLAGS.test, 'wt') as f2,
        jsonlines.Writer(f1, sort_keys=True) as w1,
        jsonlines.Writer(f2, sort_keys=True) as w2,
        smart_input(FLAGS.input, 'rt') as raw_in,
        jsonlines.Reader(raw_in) as jf):
    n1, n2 = 0, 0
    for j in jf:
      sha256 = hashlib.sha256()
      sha256.update(j['sfen'].encode('utf-8'))
      sha256.update(nonce)
      hashed = int(sha256.hexdigest(), 16)
      if hashed % 100 < FLAGS.pc:
        w1.write(j)
        n1 += 1
      else:
        w2.write(j)
        n2 += 1

  # f1.close()
  # f2.close()
  # w1.close()
  # w2.close()
  assert n1 > 0 and n2 > 0
  print(n1)
  print(n2)


if __name__ == '__main__':
  app.run(main)
