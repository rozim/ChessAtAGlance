import sys, os
import time
import collections
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorflow import keras
import pandas as pd
from data import create_input_generator, _extract, _extract2, NUM_CLASSES
from plan import load_plan



st = tf.SparseTensor(indices=[[0, 0],
                              [1, 0],
                              [1, 1]],
                     values=[0, 1, 2],
                     dense_shape=[2, 4])
print('sparse: ', st)

dense = tf.sparse.to_dense(st, default_value=-1)

print('dense: ', dense.shape, dense)

hot = tf.one_hot(dense, on_value=1.0, off_value=0.0, depth=5)
print('hot: ', hot.shape, hot)
print('hot/2: ', tf.math.reduce_sum(hot, [-2]).shape)
print('hot/2: ', tf.math.reduce_sum(hot, axis=[-2]))
print('hot/2/0: ', tf.math.reduce_sum(hot, axis=0))
print('hot/2/1: ', tf.math.reduce_sum(hot, axis=1))
print('hot/2/1b: ', tf.math.reduce_sum(hot, axis=[1]))
print('hot/2/2: ', tf.math.reduce_sum(hot, axis=2))

# print('#')
# khot = tf.keras.utils.to_categorical(dense, num_classes=5)

# print('khot: ', khot.shape, khot)

# print('k/2: ', tf.math.reduce_sum(khot, [-2]).shape)
# print('k/2: ', tf.math.reduce_sum(khot, [-2]))
