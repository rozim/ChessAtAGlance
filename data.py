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
    
  
def create_input_generator(dplan, fns, is_train=True, verbose=True):
  if type(fns) == type(""):
    fns = [fns]
  if verbose:
    print(f'Open {fns}')

  datasets = []
  for fn in fns:
    assert os.path.isfile(fn), fn
    gen1 = functools.partial(gen_snappy, fn)
    datasets.append(
      tf.data.Dataset.from_generator(gen1,
                                     output_types=('float32', 'int64'),
                                     output_shapes=(BOARD_SHAPE, [])).repeat())
    
  ds = tf.data.experimental.sample_from_datasets(
    datasets,
    weights=None # Uniform
    )
  if is_train:
    ds = ds.shuffle(dplan.shuffle)  
  ds = ds.repeat()
  ds = ds.batch(dplan.batch,
                num_parallel_calls=AUTOTUNE,
                deterministic=False) # performance
  ds = ds.prefetch(dplan.prefetch)
  return ds

    
def main(argv):
  plan = load_plan('v0.toml')
  print(next(iter(create_input_generator(plan.data, 'mega-v2-9.snappy'))))

    
if __name__ == '__main__':
  app.run(main)    
