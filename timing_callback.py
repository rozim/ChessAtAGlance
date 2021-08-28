from tensorflow import keras
import time
from absl import app
import collections
import sys, os
from tensorflow.keras.callbacks import Callback


class TimingCallback(Callback):
  def __init__(self, stateful_metrics=None):
    super(TimingCallback, self).__init__()
    self.t = {}
    self.tot = collections.defaultdict(float)

  def on_train_batch_begin(self, batch, logs=None):
    self._begin('train_batch')

  def on_train_batch_end(self, batch, logs=None):
    self._end('train_batch')

  def on_test_batch_begin(self, batch, logs=None):
    self._begin('test_batch')

  def on_test_batch_end(self, batch, logs=None):
    self._end('test_batch')

  def on_predict_batch_begin(self, batch, logs=None):
    self._begin('predict_batch')

  def on_predict_batch_end(self, batch, logs=None):
    self._end('predict_batch')

  #####


  def on_epoch_begin(self, epoch, logs=None):
    self._begin('epoch')

  def on_train_begin(self, logs=None):
    self._begin('train')
    #print('xxx', sys._getframe(  ).f_code.co_name)

  def on_test_begin(self, logs=None):
    self._begin('test')

  def on_predict_begin(self, logs=None):
    self._begin('predict')

  def on_epoch_end(self, epoch, logs=None):
    self._end('epoch')

  def on_train_end(self, logs=None):
    self._end('train')

  def on_test_end(self, logs=None):
    self._end('test')

  def on_predict_end(self, logs=None):
    self._end('predict')

  #
  def _begin(self, what):
    self.t[what] = time.perf_counter()

  def _end(self, what):
    self.tot[what] += (time.perf_counter() - self.t[what])



def main(argv):
  cb = TimingCallback()
  cb.on_train_begin()
  cb.on_train_end()
  print(cb.tot)


if __name__ == '__main__':
  app.run(main)
