from absl import app
import sys, os
import functools

import tensorflow as tf
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
from tensorflow.keras.optimizers import Adam, Ftrl
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.python.keras import backend

from plan import load_plan
from data import *

def create_model(mplan):
  kernel_regularizer = regularizers.l2(mplan.l2)
  data_format = 'channels_last'

  my_conv2d = functools.partial(
    Conv2D,
    filters=mplan.num_filters,
    kernel_size=(3, 3),
    kernel_regularizer=kernel_regularizer,
    data_format=data_format,
    padding='same',
    use_bias=False)

  my_dense = functools.partial(Dense,
                               kernel_regularizer=kernel_regularizer)

  my_bn = functools.partial(BatchNormalization, momentum=mplan.bn_momentum)

  my_activation = functools.partial(Activation, mplan.activation)
  my_dropout = functools.partial(Dropout, mplan.dropout)

  board = Input(shape=BOARD_SHAPE, dtype='float32')
  x = board
  # in: bs, chan, x, y
  #         20    8, 8
  #     0   1     2  3
  x = Permute([2, 3, 1])(x)
  # out: bs, x, y, chan
  #          8, 8, 20

  # tbd project if skip conn
  if 'num_cnn' in mplan:
    for i in range(mplan.num_cnn):
      x = my_conv2d(name=f'cnn_{i}')(x)
      x = my_bn(name=f'bn_{i}')(x)
      x = my_activation(name=f'act_{i}')(x)
      if mplan.dropout > 0.0:
        x = my_dropout(name=f'drop_{i}')(x)
  else:
    # Project to right size so skip connections work.
    x = my_conv2d(name=f'cnn_project')(x)
    x = my_bn(name=f'bn_project')(x)
    x = my_activation(name=f'act_project')(x)
    if mplan.dropout > 0.0:
      x = my_dropout(name=f'drop_project')(x)

    for i in range(mplan.num_resnet_cnn):
      skip = x
      x = my_conv2d(name=f'cnn_{i}a')(x)
      x = my_bn(name=f'bn_{i}a')(x)
      x = my_activation(name=f'act_{i}a')(x)
      if mplan.dropout > 0.0:
        x = my_dropout(name=f'drop_{i}a')(x)

      x = my_conv2d(name=f'cnn_{i}b')(x)
      x = my_bn(name=f'bn_{i}b')(x)
      x = Add(name='skip_{}b'.format(i))([x, skip])
      x = my_activation(name=f'act_{i}b')(x)
      if mplan.dropout > 0.0:
        x = my_dropout(name=f'drop_{i}b')(x)

  if mplan.do_flatten1x1:
    x = Conv2D(
      filters=mplan.num_filters,
      kernel_size=(1, 1),
      kernel_regularizer=kernel_regularizer,
      data_format=data_format,
      padding='same',
      use_bias=False,
      name='cnn_flatten')(x)
    x = my_bn(name=f'bn_flatten')(x)
    x = my_activation(name=f'act_flatten')(x)
    if mplan.dropout > 0.0:
      x = my_dropout(name=f'drop_flatten')(x)

  x = Flatten()(x)

  for i, w in enumerate(mplan.top_tower):
    x = my_dense(w, name=f'top_{i}')(x)
    x = my_bn(name=f'top_bn_{i}')(x)
    x = my_activation(name=f'top_act_{i}')(x)
    if mplan.dropout > 0.0:
      x = my_dropout(name=f'top_drop_{i}')(x)

  x = my_dense(NUM_CLASSES, name='logits', activation=None)(x)

  return Model(inputs=[board], outputs=x)



def main(_argv):

  plan = load_plan('v0.toml')
  m = create_model(plan.model)
  m.summary()

if __name__ == '__main__':
  app.run(main)
