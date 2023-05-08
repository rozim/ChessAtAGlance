import os
import os.path
import sys
import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

LOG_DIR = '/tmp/logs'

def df_to_csv(df, fn, float_format='%6.4f'):
  df.to_csv(fn, index=False, float_format=float_format)


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


class LogLrCallback(Callback):
  def __init__(self):
    super().__init__()

  def on_epoch_begin(self, epoch, logs):
    self.start_epoch = time.time()
    self.tot_train = 0.0
    self.tot_test = 0.0

  def on_epoch_end(self, epoch, logs):
    lr = self.model.optimizer.learning_rate(epoch).numpy()
    assert tf.summary.scalar('xxx_learning_rate', lr, epoch)
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


class LogTimeCallback(Callback):
  def __init__(self):
    super().__init__()

  def on_epoch_begin(self, epoch, logs):
    self.start_epoch = time.time()
    self.tot_train = 0.0
    self.tot_test = 0.0

  def on_epoch_end(self, epoch, logs):
    assert tf.summary.scalar('time/train', self.tot_train, epoch)
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
