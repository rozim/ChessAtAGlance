from contextlib import redirect_stdout
import os
import os.path
import sys
import time
import warnings

from absl import app
from absl import flags
from absl import logging

import numpy as np

import pandas as pd

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras import backend as K

import toml

#

from data import create_dataset, split_dataset
from model import create_model
from model import create_bias_only_model
from model import create_simple_model
from plan import load_plan
from lr import create_warm_linear_schedule

from tf_utils_callbacks.callbacks import BestNModelCheckpoint

#### not needed - in absl.logging
####flags.DEFINE_string('log_dir', '/tmp/logs', 'Where to write to')
flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_boolean('force_cpu', False, '')
flags.DEFINE_boolean('force_gpu', False, '')

flags.mark_flags_as_required(['plan'])
FLAGS = flags.FLAGS

T_START = time.time()

LOG_DIR = '/tmp/logs'

#####


def df_to_csv(df, fn, float_format='%6.4f'):
  df.to_csv(fn, index=False, float_format=float_format)


class LogLrCallback(Callback):
  def on_epoch_begin(self, epoch, logs):
    self.start_epoch = time.time()
    self.tot_train = 0.0
    self.tot_test = 0.0

  def on_epoch_end(self, epoch, logs):
    tf.summary.scalar('learning_rate', self.model.optimizer.lr, epoch)
    tf.summary.scalar('time/train', self.tot_train, epoch)
    tf.summary.scalar('time/test', self.tot_test, epoch)
    tf.summary.scalar('time/epoch', time.time() - self.start_epoch, epoch)

  def on_train_batch_begin(self, batch, logs=None):
    self.t_train = time.time()

  def on_train_batch_end(self, batch, logs=None):
    self.tot_train += time.time() - self.t_train

  def on_test_batch_begin(self, batch, logs=None):
    self.t_test = time.time()

  def on_test_batch_end(self, batch, logs=None):
    self.tot_test += time.time() - self.t_test


def create_log_dir(plan_fn):
  """Create unique dir."""
  base = os.path.splitext(os.path.basename(plan_fn))[0]
  try_n = 0
  while True:
    try_n += 1
    maybe = os.path.join(LOG_DIR, f'{base}_try_{try_n:02d}')
    try:
      os.makedirs(maybe)
      return maybe
    except FileExistsError:
      pass  # Dup, try next


def save_history(log_dir, df):
  fn = os.path.join(log_dir, 'history.csv')
  print(f'Write {fn}')
  with open(fn, 'w') as f:
    df_to_csv(df, f)
  os.chmod(fn, 0o444)

def save_report(log_dir, df):
  v1 = df['val_accuracy'].max()
  v2 = df['val_accuracy'].values[-1]

  fn = os.path.join(log_dir, 'report.txt')
  print(f'Write {fn}')
  with open(fn, 'w') as f:
    print(f'val_accuracy    {v1:6.4f} (best)')
    print(f'                {v2:6.4f} (last)')
    f.write(f'val_accuracy  : {v1:6.4f} (best)\n')
    f.write(f'val_accuracy  : {v2:6.4f} (last)\n')
    # if test_ev:
    #   print(f'test_accuracy   {test_ev2["accuracy"]:6.4f} (best)')
    #   print(f'                {test_ev["accuracy"]:6.4f} (last)')
    # if test_ev:
    #   f.write(f'test_accuracy : {test_ev2["accuracy"]:6.4f} (best)\n')
    #   f.write(f'test_accuracy : {test_ev["accuracy"]:6.4f} (last)\n')

    f.write(f'time          : {int(time.time() - T_START)}\n')
  os.chmod(fn, 0o444)

def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  if FLAGS.force_cpu:
    assert not FLAGS.force_gpu

  plan = load_plan(FLAGS.plan)
  tplan = plan.train
  dplan = plan.data
  mplan = plan.model
  log_dir = create_log_dir(FLAGS.plan)
  print('log_dir: ', log_dir)

  fn = os.path.join(log_dir, os.path.basename(FLAGS.plan))
  print(f'Write {fn}')
  with open(fn, 'w') as f:
    toml.dump(plan, f)
  os.chmod(fn, 0o444)

  fns_train, fns_test = split_dataset(dplan.files)
  ds_train = create_dataset(fns_train, batch=dplan.batch, shuffle=dplan.batch * 25)
  ds_val = create_dataset(fns_test, batch=dplan.batch, shuffle=None)

  # prefetch tried here with no benefit or maybe worse
  # bs = 1024 * 1024
  # tf.data.experimental.prefetch_to_device('/gpu:0', bs))

  if mplan.type == 'bias_only':
    model = create_bias_only_model(mplan)
  elif mplan.type == 'simple':
    model = create_simple_model(mplan)
  elif mplan.type == 'cnn':
    model = create_model(mplan)

  callbacks = [TerminateOnNaN(),
               LogLrCallback()
               ]

  print('log_dir: ', log_dir)

  with open(os.path.join(log_dir, 'modelsummary.txt'), 'w') as f:
    with redirect_stdout(f):
      model.summary(expand_nested=True)
      print()
      for v in model.trainable_variables:
        print(f'{v.name:40s} | {str(v.shape):20s} | {np.prod(v.shape):8d} |')

  callbacks.append(
    TensorBoard(
      log_dir=log_dir,
      histogram_freq=10,
      write_graph=False,
      write_images=False,
      write_steps_per_second=True,
      update_freq=1))

  callbacks.append(create_warm_linear_schedule(tplan))

  callbacks.append(
    tf.keras.callbacks.ModelCheckpoint(
      filepath=os.path.join(log_dir, 'checkpoint_{epoch:04d}-{val_accuracy:.2f}'),
      monitor ='val_accuracy',
      verbose=1,
      save_weights_only=True,
      save_best_only=True
    ))

  # callbacks.append(BestNModelCheckpoint(
  #   filepath=os.path.join(log_dir, 'best-checkpoint'),
  #   monitor='val_accuracy',
  #   model='max',
  #   max_to_keep=1,
  #   save_weights_only=False,
  #   verbose=1))

  model.compile(optimizer=Adam(),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  # maybe need argmax
  # tf.keras.metrics.Precision(top_k=3, name='p_3', thresholds=0.0)])

  def run_train():
    return model.fit(x=ds_train,
                     epochs=tplan.epochs,
                     steps_per_epoch=tplan.steps_per_epoch,
                     validation_data=ds_val,
                     validation_steps=tplan.val_steps,
                     callbacks=callbacks)

  if FLAGS.force_cpu:
    with tf.device('/cpu:0'):
      print('In force CPU block')
      history = run_train()
  elif FLAGS.force_gpu:
    with tf.device('/gpu:0'):
      print('In force GPU block - metal?')
      history = run_train()
  else:
    history = run_train()

  df = pd.DataFrame(history.history)
  save_history(log_dir, df)
  save_report(log_dir, df)


if __name__ == "__main__":
  app.run(main)
