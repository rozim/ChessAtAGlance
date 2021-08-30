import tensorflow as tf
import time
import functools
import sys, os, time
from absl import app
from absl import flags
from data import create_input_generator
from plan import load_plan

BOARD_SHAPE = (20, 8, 8)
BOARD_FLOATS = 1280
AUTOTUNE = tf.data.AUTOTUNE

FEATURES = {
  'board': tf.io.FixedLenFeature(BOARD_SHAPE, tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64)
}

FN1 = 'mega-v3-0.recordio'

FLAGS = flags.FLAGS

flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_integer('n', 10, '')
flags.DEFINE_integer('bs', 1024, '')
flags.DEFINE_integer('what', None, '')
flags.DEFINE_bool('return_legal_moves', False, '')

#@tf.function
def _extract(blob):
  t = tf.io.parse_example(blob, features=FEATURES)
  return t['board'], t['label']


def main(argv):
  t0 = time.time()
  flags.mark_flags_as_required(['plan'])
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  plan = load_plan(FLAGS.plan)

  dplan = plan.data
  dplan.batch = FLAGS.bs if FLAGS.bs else dplan.batch

  print(f'Read {dplan.train}')
  ds = create_input_generator(dplan,
                              dplan.train,
                              is_train=False,
                              verbose=False,
                              do_repeat=False,
                              return_legal_moves=FLAGS.return_legal_moves)
  good = 0
  goal = FLAGS.n
  for ent in ds:
    good += 1
    if good >= goal:
      break
  print('good: ', good)
  dt = time.time() - t0
  print(f'{dt:.2f}')


def old_main(argv):
  t1 = time.time()

  if FLAGS.what == 0:
    assert False
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
