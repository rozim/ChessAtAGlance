from absl import app
import sys, os
import functools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TerminateOnNaN, EarlyStopping, ModelCheckpoint, LambdaCallback, Callback
from tensorflow.keras.layers import BatchNormalization, LayerNormalization, Flatten, Add, Conv2D, Permute, Multiply
from tensorflow.keras.layers import Dense, Input, Embedding, Concatenate, Activation
from tensorflow.keras.layers import GaussianNoise, LeakyReLU, Softmax
from tensorflow.keras.layers.experimental.preprocessing import IntegerLookup, Discretization
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Ftrl
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.python.keras import backend

from plan import load_plan
from data import legal_moves_mask, NUM_CLASSES, BOARD_SHAPE


# https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
def squeeze_excite_block(i, in_block, ch, ratio=16):
  x = GlobalAveragePooling2D(name=f'se_global_{i}')(in_block)
  x = Dense(ch//ratio, activation='relu', name=f'se_dense_{i}a')(x)
  x = Dense(ch, activation='sigmoid', name=f'se_dense_{i}a'))(x)
  return Multiply(name=f'se_mul_{i}'))([in_block, x])


def create_model(mplan):
  kernel_regularizer = regularizers.l2(mplan.l2)
  data_format = 'channels_last'
  mask_legal_moves = mplan.mask_legal_moves

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

  if mask_legal_moves:
    legal_moves = Input(shape=[], name='legal_moves', dtype='int32', sparse=True)
    mask = tf.keras.layers.Lambda(legal_moves_mask,
                                  name='legal_moves_to_one_hot',
                                  output_shape=[NUM_CLASSES])(legal_moves)

  board = Input(shape=BOARD_SHAPE, name='board', dtype='float32')
  x = board
  # in: bs, chan, x, y
  #         20    8, 8
  #     0   1     2  3
  x = Permute([2, 3, 1])(x)
  # out: bs, x, y, chan
  #          8, 8, 20

  # Project to right size so skip connections work.
  x = my_conv2d(name=f'cnn_project')(x)
  x = my_bn(name=f'bn_project')(x)
  x = my_activation(name=f'act_project')(x)

  for i in range(mplan.num_resnet_cnn):
    skip = x
    x = my_conv2d(name=f'cnn_{i}a')(x)
    x = my_bn(name=f'bn_{i}a')(x)
    x = my_activation(name=f'act_{i}a')(x)

    x = my_conv2d(name=f'cnn_{i}b')(x)
    x = my_bn(name=f'bn_{i}b')(x)
    x = Add(name='skip_{}b'.format(i))([x, skip])
    x = my_activation(name=f'act_{i}b')(x)

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

  x = Flatten()(x)

  for i, w in enumerate(mplan.top_tower):
    x = my_dense(w, name=f'top_{i}')(x)
    x = my_bn(name=f'top_bn_{i}')(x)
    x = my_activation(name=f'top_act_{i}')(x)

  x = my_dense(NUM_CLASSES, name='logits', activation=None)(x)

  if mask_legal_moves:
    x = Multiply(name='mul_mask')([x, mask])
    return Model(inputs=[board, legal_moves], outputs=x)
  else:
    return Model(inputs=[board], outputs=x)



def main(_argv):

  plan = load_plan('v0.toml')
  m = create_model(plan.model)
  m.summary()

if __name__ == '__main__':
  app.run(main)
