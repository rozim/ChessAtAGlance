import sys, os
import time
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras
import pandas as pd
from data import create_input_generator
from plan import load_plan

FLAGS = flags.FLAGS

flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_string('model', None, '')
flags.DEFINE_string('fn', '', '')
flags.DEFINE_integer('n', 128, '')
flags.DEFINE_multi_string('d', None, 'override plan settings')

def main(_argv):
  flags.mark_flags_as_required(['plan', 'model'])
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  plan = load_plan(FLAGS.plan)

  # for over in FLAGS.d:
  #   print('over', over)
  #   assert '=' in over
  #   left, right = over.split('=')
  #   assert left in plan, plan.keys()
    
  # sys.exit(0)

  
  dplan, tplan = plan.data, plan.train

  model = tf.keras.models.load_model(FLAGS.model)

  steps = 1

  dres = collections.defaultdict(list)
  while steps <= FLAGS.n if FLAGS.n else tplan.test_steps:
    ds3 = create_input_generator(dplan, FLAGS.fn if FLAGS.fn else dplan.test, is_train=False, verbose=False)    
    t1 = time.time()
    test_ev = model.evaluate(x=ds3, return_dict=True, steps=steps, verbose=0)
    dt = time.time() - t1
    print(f'{steps} {test_ev} {int(dt)}')
    steps *= 2

    dres['dt'].append(dt)
    dres['loss'].append(test_ev['loss'])
    dres['accuracy'].append(test_ev['accuracy'])    
    
  df = pd.DataFrame.from_dict(dres)
  df.to_csv('evaluate.csv', index=False)
  

if __name__ == '__main__':
  app.run(main)