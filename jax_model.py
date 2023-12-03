import glob
import os
import sys
import time
from typing import Any, Dict
import warnings

from absl import app
from absl import flags
from absl import logging

import functools

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from flax.training import train_state

import tensorflow as tf

from ml_collections import config_dict
from ml_collections import config_flags

import optax

from encode import CNN_FEATURES, CNN_SHAPE_3D, NUM_CLASSES


AUTOTUNE = tf.data.AUTOTUNE

CONFIG = config_flags.DEFINE_config_file('config', 'config.py')

class ChessCNN(nn.Module):
  num_filters: int
  num_blocks: int
  num_top: int
  top_width: int

  @nn.compact
  def __call__(self, x):
    if self.num_blocks:
      # 3d: NDHWC
      # in: 1, 16, 8, 8
      # in: N  C   H  W
      #     0  1   2  3
      # out:
      #     N  H   W  C
      x = jnp.transpose(x, [0, 2, 3, 1])


      # Set up so skip conn works.
      x = nn.Conv(features=self.num_filters, kernel_size=(3, 3), padding='SAME')(x)

      for layer in range(self.num_blocks):
        skip = x

        x = nn.Conv(features=self.num_filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = nn.Conv(features=self.num_filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = x + skip
        x = nn.relu(x)

    x = x.reshape((x.shape[0], -1))  # flatten

    if self.num_top:
      for d in range(self.num_top):
        x = nn.Dense(self.top_width)(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

    # Prediction head
    logits = nn.Dense(NUM_CLASSES, use_bias=False)(x)
    return logits
    # move_probabilities = nn.softmax(move_logits)
    # return move_probabilities, value



def init_train_state(
    model, random_key, shape, learning_rate
) -> train_state.TrainState:
  # Initialize the Model
  variables = model.init(random_key, jnp.ones(shape))
  # Create the optimizer
  optimizer = optax.adam(learning_rate)
  # Create a State
  return train_state.TrainState.create(
    apply_fn=model.apply,
    tx=optimizer,
    params=variables['params']
  )



def compute_metrics(*, logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, jnp.ndarray]:
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'accuracy': accuracy,
  }
  return metrics


@jax.jit
def train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    label: jnp.ndarray
):
  def _loss_fn(params):
    logits = state.apply_fn({'params': params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=label)
    return loss.mean(), logits

  gradient_fn = jax.value_and_grad(_loss_fn, has_aux=True)
  (loss, logits), grads = gradient_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=label)
  metrics['loss'] = loss
  return state, metrics


@jax.jit
def test_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    label: jnp.ndarray
):
  logits = state.apply_fn({'params': state.params}, x)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=label)
  metrics = compute_metrics(logits=logits, labels=label)
  metrics['loss'] = loss
  return metrics


def accumulate_metrics(metrics):
  metrics = jax.device_get(metrics)
  return {
    k: np.mean([metric[k] for metric in metrics])
    for k in metrics[0]
  }

def create_dataset(shuffle: int, batch_size: int, pat: str) -> tf.data.Dataset:
  assert shuffle >= 0
  assert batch_size > 0
  assert pat

  files = glob.glob(pat)
  assert len(files) > 0, [pat, glob.glob(pat)]
  ds = tf.data.TFRecordDataset(files, 'ZLIB', num_parallel_reads=4)
  if shuffle:
    ds = ds.shuffle(shuffle)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.repeat()
  # 'board', 'label'
  ds = ds.map(functools.partial(tf.io.parse_example, features=CNN_FEATURES),
              num_parallel_calls=AUTOTUNE, deterministic=False)
  ds = ds.prefetch(AUTOTUNE)
  ds = ds.as_numpy_iterator()
  return ds


def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  config = CONFIG.value
  print(config)

  rng = jax.random.PRNGKey(int(time.time()))
  x = jnp.ones((1,) + CNN_SHAPE_3D)
  model = ChessCNN(**config.model)

  params = model.init(rng, x)
  with open('model-tabulate.txt', 'w') as f:
    f.write(model.tabulate(rng, x, console_kwargs={'width': 120}))
    flattened, _ = jax.tree_util.tree_flatten_with_path(params)
    for key_path, value in flattened:
      f.write(f'{jax.tree_util.keystr(key_path):40s} {str(value.shape):20s} {str(value.dtype):10s}\n')

  #print("P", params['params']['Dense_0'].keys())

  jax.tree_map(lambda x: x.shape, params) # Check the parameters

  state = init_train_state(
    model,
    rng,
    (config.batch_size,) + CNN_SHAPE_3D,
    config.lr,
  )

  train_iter = iter(create_dataset(**config.train.data))
  test_iter = iter(create_dataset(**config.test.data))

  t1 = time.time()
  for epoch in range(config.epochs):
    # Train
    train_metrics = []
    for i in range(config.train.steps):
      batch = next(train_iter)
      state, metrics = train_step(state, batch['board'], batch['label'])
      train_metrics.append(metrics)

    metrics = accumulate_metrics(train_metrics)
    dt = time.time() - t1
    loss = jnp.asarray(metrics['loss'])
    acc = jnp.asarray(metrics['accuracy'])
    print(f'train/{epoch:8d} {dt:6.1f}s loss={loss:6.4f} acc={acc:.4f}')

    # Test
    if config.test.steps:
      test_metrics = []
      for i in range(config.test.steps):
        batch = next(test_iter)
        metrics = test_step(state, batch['board'], batch['label'])
        test_metrics.append(metrics)

      metrics = accumulate_metrics(test_metrics)
      dt = time.time() - t1
      loss = jnp.asarray(metrics['loss'])
      acc = jnp.asarray(metrics['accuracy'])
      print(f'test/ {epoch:8d} {dt:6.1f}s loss={loss:6.4f} acc={acc:.4f}')


  print('OK')

if __name__ == "__main__":
  app.run(main)
