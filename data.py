
import glob
import os.path
import random
import sys

import tensorflow as tf
from absl import app
from encode import CNN_FEATURES, NUM_CLASSES, CNN_SHAPE_3D

AUTOTUNE = tf.data.AUTOTUNE

def extract(blob):
  t = tf.io.parse_example(blob, features=CNN_FEATURES)
  return t['board'], t['label']  # features, label [, weights]


def split_dataset(pats):
  num_files = 0
  fns = []

  for pat in pats:
    print('pat: ', pat)
    for fn in glob.glob(pat):
      num_files += 1
      assert os.path.isfile(fn), fn
      fns.append(fn)
  if num_files == 0:
    assert False, 'no files ' + pat

  random.shuffle(fns)
  split_point = int(len(fns) * 0.9)
  print('sp', split_point, len(fns))
  return fns[0:split_point], fns[split_point:]


def create_dataset(pats, batch=16, shuffle=None, repeat=True):
  num_files = 0

  for pat in pats:
    for fn in glob.glob(pat):
      num_files += 1
      assert os.path.isfile(fn), fn
  if num_files == 0:
    assert False, 'no files ' + pat

  # Don't do all else we get repeated batches it seems.
  fns_shuffle = max(1, num_files // 10)
  fns = tf.data.Dataset.list_files(pats).cache().shuffle(fns_shuffle)
  ds = tf.data.TFRecordDataset(fns, 'ZLIB', num_parallel_reads=fns_shuffle)
  if repeat:
    ds = ds.repeat()
  if shuffle:
    ds = ds.shuffle(shuffle)
  ds = ds.batch(batch, drop_remainder=False)
  ds = ds.map(extract, num_parallel_calls=AUTOTUNE, deterministic=False)
  ds = ds.prefetch(AUTOTUNE)
  return ds


def main(argv):
  pat = 'data/f1000.rio-000??-of-00100'
  goal = len(glob.glob(pat))
  train, test = split_dataset(['data/f1000.rio-000??-of-00100'])
  train_s = set(train)
  test_s = set(test)
  for ent in train:
    print('train: ', ent)
  print()
  for ent in test:
    print('test: ', ent)
  print(len(train_s))
  print(len(test_s))
  assert len(train_s & test_s) == 0
  assert len(train_s) + len(test_s) == goal

  sys.exit(0)
  ds = create_dataset('f10.rio-00009-of-00010', batch=1)
  for ent in ds.take(1):
    print(ent)
    break

if __name__ == "__main__":
  app.run(main)
