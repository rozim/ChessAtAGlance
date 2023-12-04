import glob
import os
import sys
import time
from typing import Any, Callable, Dict
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
from tensorboard.plugins.hparams import api as hp

from ml_collections import config_dict
from ml_collections import config_flags

import optax

from encode import CNN_FEATURES, CNN_SHAPE_3D, NUM_CLASSES


AUTOTUNE = tf.data.AUTOTUNE

CONFIG = config_flags.DEFINE_config_file('config', 'config.py')

LOGDIR = flags.DEFINE_string('logdir', '/tmp/logdir', '')

START = int(time.time())

ACTIVATIONS = {
  'celu': nn.celu,
  'elu': nn.elu,
  'gelu': nn.gelu,
  'glu': nn.glu,
  'hard_sigmoid': nn.hard_sigmoid,
  'hard_silu': nn.hard_silu,
  'relu': nn.relu,
  'relu6': nn.relu6,
  'selu': nn.selu,
  'sigmoid': nn.sigmoid,
  'silu': nn.silu,
  'soft_sign': nn.soft_sign,
  'softplus': nn.softplus,
}

OPTIMIZERS = {
  'adabelief': optax.adabelief,
  'adagrad': optax.adagrad,
  'adam': optax.adam,
  'adamax': optax.adamax,
  'adamaxw': optax.adamaxw,
  'adamw': optax.adamw,
  'amsgrad': optax.amsgrad,
  'dpsgd': optax.dpsgd,
  'fromage': optax.fromage,
  'lamb': optax.lamb,
  'lars': optax.lars,
  'lion': optax.lion,
  'noisy_sgd': optax.noisy_sgd,
  'novograd': optax.novograd,
  'optimistic_gradient_descent': optax.optimistic_gradient_descent,
  'radam': optax.radam,
  'rmsprop': optax.rmsprop,
  'sgd': optax.sgd,
  'sm3': optax.sm3,
  'yogi': optax.yogi,
}


class BiasOnlyDense(nn.Module):
  features: int
  param_dtype: nn.linear.Dtype = jnp.float32
  bias_init: Callable[
    [nn.linear.PRNGKey, nn.linear.Shape, nn.linear.Dtype], nn.linear.Array
  ] = nn.initializers.zeros_init()

  @nn.compact
  def __call__(self, inputs: nn.linear.Array) -> nn.linear.Array:
    bias = self.param(
        'bias', self.bias_init, (self.features,), self.param_dtype
    )
    return jnp.reshape(bias, (1, -1))

  #y = inputs
  #return jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))


class ChessCNN(nn.Module):
  num_filters: int
  num_blocks: int
  num_top: int
  top_width: int
  activation: str

  @nn.compact
  def __call__(self, x):

    f_act = ACTIVATIONS[self.activation]
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
        x = f_act(x)

        x = nn.Conv(features=self.num_filters, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.LayerNorm()(x)
        x = x + skip
        x = f_act(x)

    x = x.reshape((x.shape[0], -1))  # flatten

    if self.num_top:
      for d in range(self.num_top):
        x = nn.Dense(self.top_width)(x)
        x = nn.LayerNorm()(x)
        x = f_act(x)

    # Prediction head
    logits = nn.Dense(NUM_CLASSES, use_bias=False)(x)
    return logits
    # move_probabilities = nn.softmax(move_logits)
    # return move_probabilities, value



def init_train_state(
    model, optimizer, random_key, shape
) -> train_state.TrainState:
  # Initialize the Model
  variables = model.init(random_key, jnp.ones(shape))

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


def write_metrics(writer: tf.summary.SummaryWriter,
                  step: int,
                  metrics: Any,
                  hparams: Any = None) -> None:
  with writer.as_default(step):
    for k, v in metrics.items():
      tf.summary.scalar(k, v)
    if hparams:
      hp.hparams(hparams)
  writer.flush()



def main(argv):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  config = CONFIG.value
  assert config.model_type in ['cnn', 'bias']

  try:
    os.mkdir(LOGDIR.value)
  except:
    pass


  with open(os.path.join(LOGDIR.value, 'config.txt'), 'w') as f:
            f.write(str(config))

  rng = jax.random.PRNGKey(int(time.time()))
  x = jnp.ones((1,) + CNN_SHAPE_3D)

  if config.model_type == 'cnn':
    model = ChessCNN(**config.model)
  elif config.model_type == 'bias':
    model = BiasOnlyDense(NUM_CLASSES)

  params = model.init(rng, x)
  with open(os.path.join(LOGDIR.value, 'model-tabulate.txt'), 'w') as f:
    f.write(model.tabulate(rng, x, console_kwargs={'width': 120}))
    flattened, _ = jax.tree_util.tree_flatten_with_path(params)
    for key_path, value in flattened:
      f.write(f'{jax.tree_util.keystr(key_path):40s} {str(value.shape):20s} {str(value.dtype):10s}\n')

  jax.tree_map(lambda x: x.shape, params) # Check the parameters
  del params

  if config.train.optimizer == 'lion':
    optimizer = OPTIMIZERS[config.train.optimizer](**config.optimizer.lion)
    print("LION: ", optimizer)
  else:
    optimizer = OPTIMIZERS[config.train.optimizer](config.train.lr),
  state = init_train_state(
    model,
    optimizer,
    rng,
    (config.batch_size,) + CNN_SHAPE_3D,
  )

  train_iter = iter(create_dataset(**config.train.data))
  test_iter = iter(create_dataset(**config.test.data))

  train_writer = tf.summary.create_file_writer(
    os.path.join(LOGDIR.value, 'train'))
  test_writer = tf.summary.create_file_writer(
    os.path.join(LOGDIR.value, 'test'))
  final_writer = tf.summary.create_file_writer(
    os.path.join(LOGDIR.value, 'final'))

  hparams = {
    'start': (START - 1701632426),
    'lr': config.train.lr,
    'num_blocks': config.model.num_blocks,
    'num_filters': config.model.num_filters,
    'num_top': config.model.num_top,
    'top_width': config.model.top_width,
    'activation': config.model.activation,
    'optimizer': config.train.optimizer,
  }
  if config.train.optimizer == 'lion':
    hparams['lion.b1'] = config.optimizer.lion.b1
    hparams['lion.b2'] = config.optimizer.lion.b2
    hparams['lion.weight_decay'] = config.optimizer.lion.weight_decay


  for epoch in range(config.epochs):
    t1 = time.time()
    # Train
    train_metrics = []
    for i in range(config.train.steps):
      batch = next(train_iter)
      state, metrics = train_step(state, batch['board'], batch['label'])
      train_metrics.append(metrics)

    metrics = accumulate_metrics(train_metrics)
    dt = time.time() - t1
    train_loss = jnp.asarray(metrics['loss'])
    train_acc = jnp.asarray(metrics['accuracy'])
    print(f'train/{epoch:8d} {dt:6.1f}s loss={train_loss:6.4f} acc={train_acc:.4f}')

    write_metrics(train_writer, epoch,
                  {'accuracy': train_acc,
                   'loss': train_loss,
                   'time/elapsed': dt,
                   'time/xps': (config.train.steps * config.train.data.batch_size) / dt})


    # Test
    if config.test.steps:
      t1 = time.time()
      test_metrics = []
      for i in range(config.test.steps):
        batch = next(test_iter)
        metrics = test_step(state, batch['board'], batch['label'])
        test_metrics.append(metrics)

      metrics = accumulate_metrics(test_metrics)
      dt = time.time() - t1
      test_loss = jnp.asarray(metrics['loss'])
      test_acc = jnp.asarray(metrics['accuracy'])
      print(f'test/ {epoch:8d} {dt:6.1f}s loss={test_loss:6.4f} acc={test_acc:.4f}')
      write_metrics(test_writer, epoch,
                    {'accuracy': test_acc,
                     'loss': test_loss,
                     'time/elapsed': dt,
                     'time/xps': (config.test.steps * config.test.data.batch_size) / dt
                     })


    write_metrics(final_writer, config.epochs,
                  {
                    'train/accuracy': train_acc,
                   'train/loss': train_loss,
                    'test/accuracy': test_acc,
                   'test/loss': test_loss,
                   },
                  hparams)


  if config.model_type == 'bias':
    from encode_move import INDEX_TO_MOVE
    with open(os.path.join(LOGDIR.value, 'bias.txt'), 'w') as f:
      soft = jax.nn.softmax(state.params['bias'])
      besti = jnp.argmax(soft, -1)
      for i, foo in enumerate(soft.tolist()):
        f.write(f'{i:4d} {INDEX_TO_MOVE[i]:6s} {100.0*foo:6.4f}\n')



if __name__ == "__main__":
  app.run(main)
