
import glob
import os.path
import sys

import tensorflow as tf
from absl import app
from encode import CNN_FEATURES, NUM_CLASSES, CNN_SHAPE_3D

AUTOTUNE = tf.data.AUTOTUNE

def extract(blob):
  t = tf.io.parse_example(blob, features=CNN_FEATURES)
  return t['board'], t['label']  # features, label [, weights]


def create_dataset(pats, batch=16, shuffle=None, repeat=True):
  num_files = 0

  for pat in pats:
    for fn in glob.glob(pat):
      num_files += 1
      assert os.path.isfile(fn), fn
  if num_files == 0:
    assert False, 'no files ' + pat

  fns = tf.data.Dataset.list_files(pats).cache().shuffle(num_files)
  ds = tf.data.TFRecordDataset(fns, 'ZLIB', num_parallel_reads=num_files)
  if repeat:
    ds = ds.repeat()
  if shuffle:
    ds = ds.shuffle(shuffle)
  ds = ds.batch(batch, drop_remainder=False)
  ds = ds.map(extract, num_parallel_calls=AUTOTUNE, deterministic=False)
  ds = ds.prefetch(AUTOTUNE)
  return ds


def main(argv):
  ds = create_dataset('f10.rio-00009-of-00010', batch=1)
  for ent in ds.take(1):
    print(ent)
    break

if __name__ == "__main__":
  app.run(main)
