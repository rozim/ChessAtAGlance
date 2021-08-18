import sys, os
import time
import tensorflow as tf
import functools
from absl import app

from plan import *
from snappy_io import unsnappy


NUM_CLASSES = 4672
BOARD_SHAPE = (20, 8, 8)
BOARD_FLOATS = 1280

AUTOTUNE = tf.data.AUTOTUNE


def gen_snappy(fn):
  for ex in unsnappy(fn):
    board = tf.convert_to_tensor(ex.features.feature['board'].float_list.value,
                                 dtype=tf.float32)
    board = tf.reshape(board, BOARD_SHAPE)
    action = tf.convert_to_tensor(ex.features.feature['label'].int64_list.value[0],
                                 dtype=tf.int64)
    yield (board, action)
    
  
def create_input_generator(dplan, fn, is_train=True, verbose=True):
  assert os.path.isfile(fn), fn
  if verbose:
    print(f'Open {fn}')
  gen1 = functools.partial(gen_snappy, fn)
  ds1 = tf.data.Dataset.from_generator(gen1,
                                      output_types=('float32', 'int64'),
                                      output_shapes=(BOARD_SHAPE, []))
  if is_train:
    ds1 = ds1.shuffle(dplan.shuffle)
  ds1 = ds1.repeat()
  ds1 = ds1.batch(dplan.batch,
                  num_parallel_calls=AUTOTUNE,
                  deterministic=False) # performance
  ds1 = ds1.prefetch(dplan.prefetch)
  return ds1    

    
def main(argv):
  plan = load_plan('v0.toml')
  print(next(iter(create_input_generator(plan.data, 'mega-v2-9.snappy'))))

    
if __name__ == '__main__':
  app.run(main)    
