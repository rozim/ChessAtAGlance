import sys
import tensorflow as tf
import leveldb
from absl import app
from absl import flags
from absl import logging
from datetime import datetime

import glob
import toml
import re
from contextlib import redirect_stdout
import collections
import datetime
import functools
import itertools
import math
import numpy as np
import os

import random
import sys
import time

from zipfile import ZipFile
from datetime import datetime

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, LambdaCallback, Callback
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten, Add, Conv2D, Permute
from tensorflow.keras.layers import Dense, Dropout, Input, Embedding, Concatenate, Activation
from tensorflow.keras.layers import GaussianNoise, LeakyReLU, Softmax
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, Discretization
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Ftrl, SGD

from tensorflow.python.keras import backend
import pandas as pd

from data import create_input_generator
from plan import load_plan
from model import create_model
from lr import create_lr_schedule

from tf_utils_callbacks.callbacks import BestNModelCheckpoint


FLAGS = flags.FLAGS
flags.DEFINE_string('plan', None, 'toml file')

flags.DEFINE_multi_string('d', None, 'override plan settings')


def df_to_csv(df, fn, float_format='%6.4f'):
  df.to_csv(fn, index=False, float_format=float_format)


class LogLrCallback(Callback):
  def on_epoch_end(self, epoch, logs):
    logs['lr'] = backend.get_value(self.model.optimizer.lr(epoch))


def main(_argv):
  flags.mark_flags_as_required(['plan'])
  t1 = time.time()
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  out_dir = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
  out_dir = os.path.join('results', out_dir)
  print('mkdir', out_dir)
  os.mkdir(out_dir)  
  
  plan = load_plan(FLAGS.plan)
  fn = os.path.join(out_dir, FLAGS.plan)
  print(f'Write {fn}')
  with open(fn, 'w') as f:
    toml.dump(plan, f)
  os.chmod(fn, 0o444)    

  dplan = plan.data
  ds1 = create_input_generator(dplan, dplan.train, is_train=True)
  ds2 = create_input_generator(dplan, dplan.validate, is_train=False)
  ds3 = create_input_generator(dplan, dplan.test, is_train=False) if 'test' in dplan else None

  m = create_model(plan.model)
  fn = os.path.join(out_dir, 'model-summary.txt')
  print(f'Write {fn}')
  with open(fn, 'w') as f:
    with redirect_stdout(f):
      m.summary()
  os.chmod(fn, 0o444)

  tplan = plan.train
  lr = create_lr_schedule(tplan)
  # lr = CosineDecayRestarts(initial_learning_rate=tplan.lr,
  #                          first_decay_steps=tplan.first_decay_steps,
  #                          t_mul=1,
  #                          m_mul=1,
  #                          alpha=tplan.alpha)  

  if tplan.optimizer == 'SGD':
    optimizer = SGD(learning_rate=lr)
  elif tplan.optimizer == 'Adam':
    optimizer = Adam(learning_rate=lr)
  else:
    assert False, tplan.optimizer
  m.compile(optimizer=optimizer,
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
  #tf.keras.metrics.Precision(top_k=3, name='p_3'),
  #tf.keras.metrics.Recall(top_k=3, name='r_3')])

  callbacks = [TerminateOnNaN(),
               LogLrCallback()]

  best_path = os.path.join(out_dir, 'best.model')
  callbacks.append(BestNModelCheckpoint(
    filepath=best_path,
    monitor='val_accuracy',
    model='max',
    max_to_keep=1,
    save_weights_only=False,
    verbose=1))

  history = m.fit(x=ds1,
                  epochs=tplan.epochs,
                  steps_per_epoch=tplan.steps_per_epoch,
                  validation_data=ds2,
                  validation_steps=tplan.validation_steps,
                  callbacks=callbacks)
  df = pd.DataFrame(history.history)

  fn = os.path.join(out_dir, 'last.model')
  print(f'Write {fn}')    
  m.save(fn)
  os.chmod(fn, 0o755) 

  if ds3:
    tt0 = time.time()
    print('Test (last)')
    test_ev = m.evaluate(x=ds3, return_dict=True, steps=tplan.test_steps)
    dt = time.time() - tt0
    print('Test:', test_ev, int(dt))

    print('Test (best)')
    tt0 = time.time()
    ds3 = create_input_generator(dplan, dplan.test, is_train=False) # rewind    
    test_ev2 = tf.keras.models.load_model(best_path).evaluate(x=ds3, return_dict=True, steps=tplan.test_steps)
    dt = time.time() - tt0
    print('Test/2:', test_ev2, int(dt))    

  fn = os.path.join(out_dir, 'history.csv')
  print(f'Write {fn}')  
  with open(fn, 'w') as f:  
    df_to_csv(df, f)
  os.chmod(fn, 0o444)        

  v1 = df['val_accuracy'].max()
  v2 = df['val_accuracy'].values[-1]

  fn = os.path.join(out_dir, 'report.txt')
  print(f'Write {fn}')  
  with open(fn, 'w') as f:    
    print(f'val_accuracy    {v1:6.4f} (best)')
    print(f'                {v2:6.4f} (last)')
    print(f'test_accuracy   {test_ev2["accuracy"]:6.4f} (best)')        
    print(f'                {test_ev["accuracy"]:6.4f} (last)')
    
    f.write(f'val_accuracy  : {v1:6.4f} (best)\n')
    f.write(f'val_accuracy  : {v2:6.4f} (last)\n')

    f.write(f'test_accuracy : {test_ev2["accuracy"]:6.4f} (best)\n')
    f.write(f'test_accuracy : {test_ev["accuracy"]:6.4f} (last)\n')    

    f.write(f'time          : {int(time.time() - t1)}\n')    
  os.chmod(fn, 0o444) 



if __name__ == '__main__':
  app.run(main)

  
