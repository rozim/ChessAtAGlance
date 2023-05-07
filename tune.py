
from contextlib import redirect_stdout
import functools
import os
import os.path
import sys
import time
import toml
import warnings

from absl import app
from absl import flags
from absl import logging

import keras_tuner

from Mish.Mish.TFKeras import mish

import numpy as np

import pandas as pd

import smelu

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras import backend as K

from data import create_dataset, split_dataset
from model import create_model
from plan import load_plan
from lr import create_warm_linear_schedule
from train_util import df_to_csv, create_log_dir

from tf_utils_callbacks.callbacks import BestNModelCheckpoint

from train_util import df_to_csv, create_log_dir, LogLrCallback

flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_integer('verbose', 1, '')


flags.mark_flags_as_required(['plan'])
FLAGS = flags.FLAGS

T_START = time.time()

ACTIVATION = [
  'Mish',
  'elu',
  'exponential',
  'gelu',
  'hard_sigmoid',
  'linear',
  'leaky_relu',
  'log_softmax',
  'relu',
  'relu6',
  'selu',
  'silu',
  'sigmoid',
  'smelu',
  'softmax',
  'softplus',
  'softsigh',
  'swish',
  'tanh',
  ]


def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  assert tf.keras.activations.get('swish')
  assert tf.keras.activations.get('smelu')
  assert tf.keras.activations.get('Mish')
  tf.keras.layers.Activation(activation='Mish')
  tf.keras.layers.Activation(activation='smelu')
  tf.keras.layers.Dense(1, activation='Mish')
  tf.keras.layers.Dense(1, activation='smelu')

  plan = load_plan(FLAGS.plan)
  tplan = plan.train
  dplan = plan.data
  mplan = plan.model
  tune_plan = plan.tune
  log_dir = create_log_dir(FLAGS.plan)

  fns_train, fns_test = split_dataset(dplan.files)
  ds_train = create_dataset(fns_train, batch=dplan.batch, shuffle=dplan.batch * 25)
  ds_val = create_dataset(fns_test, batch=dplan.batch, shuffle=None)

  def make_build_model_fn(mplan):
    def _build_model(hp):
      mplan['activation'] = hp.Choice('activation', ACTIVATION)
      m = create_model(mplan)
      m.compile(optimizer=tf.keras.optimizers.legacy.Adam(),
                loss=SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
      return m
    return _build_model


#  tuner = keras_tuner.RandomSearch(
  tuner = keras_tuner.BayesianOptimization(
    hypermodel=make_build_model_fn(mplan),
    objective='val_accuracy',
    max_trials=tune_plan.trials,
    executions_per_trial=tune_plan.executions,
    directory=tune_plan.log_dir,
    project_name='activation',
    overwrite=True)

  tuner.search_space_summary()
  print()

  callbacks = [TerminateOnNaN(),
               LogLrCallback(),
               TensorBoard(log_dir=tune_plan.tb_dir),
               create_warm_linear_schedule(tplan)
               ]

  tuner.search(x=ds_train,
               validation_data=ds_val,
               epochs=tplan.epochs,
               steps_per_epoch=tplan.steps_per_epoch,
               validation_steps=tplan.val_steps,
               verbose=FLAGS.verbose,
               callbacks=callbacks)


  print('###')
  best_model = tuner.get_best_models()[0]
  print('### ', best_model)
  print()

  print('# HP summary')
  print(tuner.get_best_hyperparameters())

  print()
  print('# Results summary')
  tuner.results_summary(num_trials=tune_plan.trials * 2)
  print()
  print('# all done')




if __name__ == "__main__":
  app.run(main)
