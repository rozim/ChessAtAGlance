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
    self.num = collections.defaultdict(int)

  def on_train_batch_begin(self, batch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_train_batch_end(self, batch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_test_batch_begin(self, batch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_test_batch_end(self, batch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_predict_batch_begin(self, batch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_predict_batch_end(self, batch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  #####


  def on_epoch_begin(self, epoch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_train_begin(self, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_test_begin(self, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_predict_begin(self, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_epoch_end(self, epoch, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_train_end(self, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_test_end(self, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  def on_predict_end(self, logs=None):
    self._doit(sys._getframe().f_code.co_name)

  #
  # maybe more generic again is
  # https://stackoverflow.com/questions/2704434/intercept-method-calls-in-python
  def _doit(self, func_name):
    if func_name.endswith('_begin'):
      what = func_name.removesuffix('_begin')
      self.t[what] = time.perf_counter()
    else:
      assert func_name.endswith('_end')
      what = func_name.removesuffix('_end')
      assert self.t[what]
      self.tot[what] += (time.perf_counter() - self.t[what])
      self.num[what] += 1
      self.t[what] = None



def main(argv):
  cb = TimingCallback()
  cb.on_train_begin()
  cb.on_train_batch_begin(None)
  cb.on_train_batch_end(None)
  cb.on_train_batch_begin(None)
  cb.on_train_batch_end(None)
  cb.on_train_end()
  print(cb.tot)


if __name__ == '__main__':
  app.run(main)
