import sys, os
import time
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras
import pandas as pd
from data import create_input_generator, _extract, _extract2, NUM_CLASSES
from plan import load_plan

FLAGS = flags.FLAGS

flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_string('model', None, '')
flags.DEFINE_string('fn', '', '')
flags.DEFINE_integer('n', 1, '')
flags.DEFINE_integer('bs', 4, '')
flags.DEFINE_multi_string('d', None, 'override plan settings')

def main(_argv):
  t0 = time.time()
  flags.mark_flags_as_required(['plan', 'model'])
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  plan = load_plan(FLAGS.plan)

  dplan, tplan = plan.data, plan.train
  dplan.batch = FLAGS.bs if FLAGS.bs else dplan.batch

  model = tf.keras.models.load_model(FLAGS.model)

  steps = 1

  goal = FLAGS.n if FLAGS.n else tplan.test_steps
  ds = create_input_generator(dplan,
                              FLAGS.fn if FLAGS.fn else dplan.test,
                              is_train=False,
                              verbose=False,
                              do_repeat=False,
                              return_legal_moves=True)
  it = iter(ds)
  x, y = next(it)
  print('board: ', x['board'])
  print('legal_moves: ', x['legal_moves'])
  print('legal_moves_mask: ', x['legal_moves_mask'])
  print('y: ', y)



if __name__ == '__main__':
  app.run(main)
