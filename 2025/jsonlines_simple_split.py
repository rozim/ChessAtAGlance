
# Split a jsonlines file into training and test data.
# Assume it is already shuffled.
# Simplified version w/o compression from ../jsonlines_split.py

import io

from absl import app
from absl import flags
from absl import logging

import time


FLAGS = flags.FLAGS

flags.DEFINE_string('input',
                    'data/mega2600_shuffled.jsonl',
                    'Input - uncompressed json lines')
flags.DEFINE_string('train',
                    'data/mega2600_shuffled_train.json',
                    'Output - train')
flags.DEFINE_string('test',
                    'data/mega2600_shuffled_test.json',
                    'Output - test')


flags.DEFINE_integer('pc', 99, 'Percent train')


def main(argv):
  t1 = time.time()
  f_in = open(FLAGS.input, 'r')

  f1 = open(FLAGS.train, 'w')
  f2 = open(FLAGS.test, 'w')

  for i, line in enumerate(f_in):
    if i % 100 < FLAGS.pc:
      f1.write(line)
    else:
      f2.write(line)
  f1.close()
  f2.close()
  f_in.close()


if __name__ == '__main__':
  app.run(main)
