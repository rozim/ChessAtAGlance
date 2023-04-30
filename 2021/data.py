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

FEATURES = {
  'board': tf.io.FixedLenFeature(BOARD_SHAPE, tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64)
}

FEATURES2 = {
  'board': tf.io.FixedLenFeature(BOARD_SHAPE, tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'legal_moves': tf.io.VarLenFeature(tf.int64)
}


def legal_moves_mask(legal_moves_sparse):
  dense = tf.sparse.to_dense(legal_moves_sparse, default_value=-1)
  hot = tf.one_hot(dense, on_value=1.0, off_value=0.0, depth=NUM_CLASSES)
  hot2 = tf.math.reduce_sum(hot, axis=[-2])
  return hot2


def _extract(blob):
  t = tf.io.parse_example(blob, features=FEATURES)
  return t['board'], t['label']


def _extract2(blob):
  t = tf.io.parse_example(blob, features=FEATURES2)
  return ({'board': t['board'],
           'legal_moves': t['legal_moves'],
           #'legal_moves_mask': legal_moves_mask(t['legal_moves']) # very slow - try to do in model and maybe runs on gpu?
           },
          t['label'])


# def gen_snappy(fn):
#   for ex in unsnappy(fn):
#     board = tf.convert_to_tensor(ex.features.feature['board'].float_list.value,
#                                  dtype=tf.float32)
#     board = tf.reshape(board, BOARD_SHAPE)
#     action = tf.convert_to_tensor(ex.features.feature['label'].int64_list.value[0],
#                                  dtype=tf.int64)
#     yield (board, action)


def create_input_generator_rio(dplan, fns, is_train=True, verbose=True, do_repeat=True, return_legal_moves=False):
  if type(fns) == type(""):
    fns = [fns]
  if verbose:
    print(f'Open {fns}')

  datasets = []
  for fn in fns:
    assert os.path.isfile(fn), fn
    assert fn.endswith('.recordio')
  ds = tf.data.TFRecordDataset(fns, 'ZLIB', num_parallel_reads=len(fns))
  if is_train:
    ds = ds.shuffle(dplan.shuffle)
  if do_repeat:
    ds = ds.repeat()
  if dplan.get('swap_batch_map_order', False):
    ds = ds.batch(dplan.batch,
                  num_parallel_calls=AUTOTUNE,
                  deterministic=False)
    ds = ds.map(_extract2 if return_legal_moves else _extract,
                num_parallel_calls=AUTOTUNE)
  else:
    ds = ds.batch(dplan.batch,
                  num_parallel_calls=AUTOTUNE,
                  deterministic=False) # performance
    ds = ds.map(_extract2 if return_legal_moves else _extract)
  ds = ds.prefetch(dplan.prefetch)
  return ds


def create_input_generator(dplan, fns, is_train=True, verbose=True, do_repeat=True, return_legal_moves=False):
  if type(fns) == type(""):
    fns = [fns]
  if fns[0].endswith('.recordio'):
    return create_input_generator_rio(dplan, fns, is_train, verbose, do_repeat, return_legal_moves)
  assert False, 'obsolete code path'

  # if verbose:
  #   print(f'Open {fns}')

  # datasets = []
  # for fn in fns:
  #   assert os.path.isfile(fn), fn
  #   assert fn.endswith('.snappy')
  #   gen1 = functools.partial(gen_snappy, fn)

  #   ds = tf.data.Dataset.from_generator(gen1,
  #                                       output_types=('float32', 'int64'),
  #                                       output_shapes=(BOARD_SHAPE, []))
  #   if do_repeat:
  #     ds = ds.repeat()
  #   datasets.append(ds)
  #   del ds

  # ds = tf.data.experimental.sample_from_datasets(
  #   datasets,
  #   weights=None # Uniform
  #   )
  # if is_train:
  #   ds = ds.shuffle(dplan.shuffle)
  # ds = ds.repeat()
  # ds = ds.batch(dplan.batch,
  #               num_parallel_calls=AUTOTUNE,
  #               deterministic=False) # performance
  # ds = ds.prefetch(dplan.prefetch)
  # return ds


def main(argv):
  plan = load_plan('v0.toml')
  print(next(iter(create_input_generator(plan.data, 'mega-v2-9.snappy'))))


if __name__ == '__main__':
  app.run(main)
