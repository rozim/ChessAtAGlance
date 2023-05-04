
import tensorflow as tf

# Smooth activations and reproducibility in deep networks
# https://arxiv.org/abs/2010.09931

from tensorflow.keras.layers import Activation
# from tensorflow.keras.utils.generic_utils import get_custom_objects


def smelu(x, beta=1.0):
  smooth = (0.25 / beta) * x**2 + 0.5 * x * beta / 4
  y = tf.zeros_like(x)
  y = tf.where((x > -beta) & (x < beta), smooth, y)
  return tf.where(x > beta, x, y)

tf.keras.saving.get_custom_objects().update({'smelu': Activation(smelu)})
