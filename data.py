import sys, os
import leveldb
import time
import tensorflow as tf
import functools
from absl import app

from plan import *


NUM_CLASSES = 4672
BOARD_SHAPE = (20, 8, 8)
BOARD_FLOATS = 1280

AUTOTUNE = tf.data.AUTOTUNE


def gen(fn):
  db = leveldb.LevelDB(fn)
  for ent in db.RangeIter():
    ex = tf.train.Example().FromString(ent[1])
    board = tf.convert_to_tensor(ex.features.feature['board'].float_list.value,
                                 dtype=tf.float32)
    board = tf.reshape(board, BOARD_SHAPE)
    action = tf.convert_to_tensor(ex.features.feature['label'].int64_list.value[0],
                                 dtype=tf.int64)
    yield (board, action)

    
def create_input_generator(dplan, fn, is_train=True, verbose=True):
  assert os.path.isdir(fn), fn
  if verbose:
    print(f'Open {fn}')
  gen1 = functools.partial(gen, fn)
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
  #plan = load_plan('v0.toml')
  #print(next(iter(create_input_generator(plan))))
  #sys.exit(0)
  
  row = 0
  t1 = time.time()
  mod = 1
  for ent in gen('mega-v2-8.leveldb'):
    row += 1
    if row % mod == 0:
      print(row, int(time.time() - t1))
      mod *= 2

  print('done: ', row, 'rows')
  print('done: ', int(time.time() - t1), 's')

  
    
if __name__ == '__main__':
  app.run(main)    
