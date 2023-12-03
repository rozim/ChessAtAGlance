import glob
import os
import sys
import time
from typing import Any, Dict

from absl import app
from absl import flags
from absl import logging

import functools

import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.training import train_state

import tensorflow as tf

from ml_collections import config_dict

import optax

from encode import CNN_FEATURES, CNN_SHAPE_3D, NUM_CLASSES


AUTOTUNE = tf.data.AUTOTUNE


class ChessCNN(nn.Module):
  num_filters: int
  num_layers: int

  @nn.compact
  def __call__(self, x):
    # 3d: NDHWC
    # in: 1, 16, 8, 8
    # in: N  C   H  W
    #     0  1   2  3
    # out:
    #     N  H   W  C
    x = jnp.transpose(x, [0, 2, 3, 1])
    for layer in range(self.num_layers):
      x = nn.Conv(features=self.num_filters, kernel_size=(3, 3), padding='SAME')(x)
      x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten

    # Prediction head
    logits = nn.Dense(NUM_CLASSES)(x)
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



def train_and_evaluate(train_dataset, eval_dataset, test_dataset, state, epochs):
  pass


def compute_metrics(logits: jnp.ndarray,
                    labels: jnp.ndarray) -> Any:
  pass

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
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=label)
    return loss.mean(), logits


  gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = gradient_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(logits=logits, labels=label)
  metrics['loss'] = loss
  return state, metrics




def main(argv):
  rng = jax.random.PRNGKey(0)
  x = jnp.ones((1,) + CNN_SHAPE_3D)
  model = ChessCNN(num_filters=64, num_layers=1)
  params = model.init(rng, x)
  jax.tree_map(lambda x: x.shape, params) # Check the parameters

  lr = 5e-3
  batch_size = 1024
  epochs = 1000
  mod = 10

  state = init_train_state(
    model,
    rng,
    (batch_size,) + CNN_SHAPE_3D,
    lr,
  )
  files = glob.glob('data/cnn-1m-?????-of-00010.recordio')
  ds = tf.data.TFRecordDataset(files, 'ZLIB', num_parallel_reads=4)
  ds = ds.shuffle(batch_size * 10)
  ds = ds.batch(batch_size, drop_remainder=True)
  ds = ds.repeat()
  # 'board', 'label'
  ds = ds.map(functools.partial(tf.io.parse_example, features=CNN_FEATURES),
              num_parallel_calls=AUTOTUNE, deterministic=False)
  ds = ds.prefetch(AUTOTUNE)
  ds = ds.as_numpy_iterator()
  #ds = ds.map(convert_to_jax)
  ds_iter = iter(ds)

  t1 = time.time()
  for i in range(epochs):
    batch = next(ds_iter)
    state, metrics = train_step(state, batch['board'], batch['label'])
    if i % mod == 0:
      dt = time.time() - t1
      loss = jnp.asarray(metrics['loss'])
      acc = jnp.asarray(metrics['accuracy'])
      print(f'{i:8d} {dt:6.1f}s loss={loss:.4f} acc={acc:.4f}')



  # print(next(ds_iter))
  # return
  # for ent in iter(ds):
  #   print(ent)
  #   break
  #   #print(model.apply(ent['board']))
  print('OK')
if __name__ == "__main__":
  app.run(main)
