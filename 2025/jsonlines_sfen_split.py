
# Split a jsonlines file into training and test data.
# Assume it is already shuffled.
# Simplified version w/o compression from ../jsonlines_split.py

import hashlib
import os
import random
import sys
import time

import jsonlines
from absl import app, flags, logging

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


def main(argv):
  assert os.path.exists(FLAGS.input)
  assert not os.path.exists(FLAGS.train)
  assert not os.path.exists(FLAGS.test)
  assert FLAGS.train != FLAGS.test

  nonce = str(random.random()).encode('utf-8')

  f1 = open(FLAGS.train, 'w')
  f2 = open(FLAGS.test, 'w')

  w1 = jsonlines.Writer(f1, sort_keys=True)
  w2 = jsonlines.Writer(f2, sort_keys=True)
  n1, n2 = 0, 0
  for j in jsonlines.open(FLAGS.input, 'r'):
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

  f1.close()
  f2.close()
  w1.close()
  w2.close()
  assert n1 > 0 and n2 > 0
  print(n1)
  print(n2)


if __name__ == '__main__':
  app.run(main)
