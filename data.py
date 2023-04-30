import glob
import tensorflow as tf
from absl import app
from encode import CNN_FEATURES, NUM_CLASSES, CNN_SHAPE_3D

AUTOTUNE = tf.data.AUTOTUNE

def extract(blob):
  t = tf.io.parse_example(blob, features=CNN_FEATURES)
  return t['board'], t['label']


def create_dataset(pat, batch=16, shuffle=None, repeat=True):
  fns = tf.data.Dataset.list_files(pat)
  ds = tf.data.TFRecordDataset(fns, 'ZLIB', num_parallel_reads=len(fns))
  if repeat:
    ds = ds.repeat()
  if shuffle:
    ds = ds.shuffle(shuffle)
  ds = ds.batch(batch,drop_remainder=False)
  ds = ds.map(extract, num_parallel_calls=AUTOTUNE, deterministic=False)
  ds = ds.prefetch(AUTOTUNE)
  return ds


def main(argv):
  ds = create_dataset(['foo3.rio-0000?-of-00010'], batch=1)
  for ent in ds.take(1):
    print(ent)
    break

if __name__ == "__main__":
  app.run(main)
