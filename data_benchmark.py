import tensorflow as tf
import time
import functools
import sys, os, time
from absl import app
from absl import flags
from data import create_input_generator, gen_snappy

BOARD_SHAPE = (20, 8, 8)
BOARD_FLOATS = 1280
AUTOTUNE = tf.data.AUTOTUNE

FEATURES = {
  'board': tf.io.FixedLenFeature(BOARD_SHAPE, tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64)
}

FN0 = 'mega-v3-0.snappy'
FN1 = 'mega-v3-0.recordio'


FLAGS = flags.FLAGS

flags.DEFINE_integer('n', 10, '')
flags.DEFINE_integer('bs', 1024, '')
flags.DEFINE_integer('what', None, '')

#@tf.function
def _extract(blob):
  t = tf.io.parse_example(blob, features=FEATURES)
  return t['board'], t['label']


def main(argv):
  t1 = time.time()

  if FLAGS.what == 0:
    print('DS: old')
    gen1 = functools.partial(gen_snappy, FN0)
    ds = tf.data.Dataset.from_generator(gen1,
                                        output_types=('float32', 'int64'),
                                        output_shapes=(BOARD_SHAPE, []))
    ds = ds.shuffle(1024)
    ds = ds.batch(FLAGS.bs)
  else:
    assert FLAGS.what == 1
    print('DS: new')
    ds = tf.data.TFRecordDataset([FN1], 'ZLIB') # buffer_size doesn't help
    ds = ds.shuffle(1024)
    ds = ds.batch(FLAGS.bs)
    ds = ds.map(_extract) # no benefit: num_parallel_calls=4, deterministic=False)

  ds = ds.prefetch(2) # value doesn't seem to matter, even 0
  first = True
  for batch in iter(ds.take(FLAGS.n)):
   if first:
     print(type(batch), len(batch))
     print(batch[0].shape)
     first = False
  dt = time.time() - t1
  print(int(dt))
  print('ex/sec: ', (FLAGS.n * FLAGS.bs) / dt)






if __name__ == '__main__':
  app.run(main)
