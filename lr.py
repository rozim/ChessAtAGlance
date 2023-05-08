import functools
import sys

from absl import app
from plan import *


import tensorflow as tf
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

def create_lr_scheduler(tplan):
  """ Returns callback"""
  if tplan.lr_schedule == 'warm_linear':
    return (create_warm_linear_scheduler(tplan), tplan.lr)
  assert tplan.lr_schedule == 'cosine'

  lr = CosineDecayRestarts(initial_learning_rate=tplan.lr,
                           first_decay_steps=tplan.first_decay_steps,
                           t_mul=1,
                           m_mul=1,
                           alpha=tplan.alpha)
  return lr



def create_warm_linear_scheduler(tplan):
  def _create_warm_linear_scheduler(epoch, ignore_lr):
    lr, train_epochs, train_warmup = (tplan.lr,
                                      tplan.epochs,
                                      tplan.warmup)

    epoch = epoch + 1.0

    if epoch <= train_warmup:
      return lr * (epoch / train_warmup)
    else:
      pct = (epoch - train_warmup) / (train_epochs - train_warmup)
      pct *= tplan.get('lr_max_decay_factor', 1.0)
      return lr - (pct * lr)
  return LearningRateScheduler(_create_warm_linear_scheduler)


def create_poly_schedule(tplan):
  return tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=tplan.lr,
    decay_steps=tplan.epochs * tplan.steps_per_epoch,
    end_learning_rate=tplan.lr * tplan.lr_max_decay_factor,
    power=1.0, # linear
  )


def create_warm_schedule(tplan):
  max_steps = tplan.epochs * tplan.steps_per_epoch
  warmup_steps = tplan.warmup_steps
  max_decay_factor = tplan.get('lr_max_decay_factor', 1.0)
  lr = tplan.lr
  values = []

  for step in range(1, max_steps + 1):
    if step <= warmup_steps:
      values.append(lr * (step / warmup_steps))
    else:
      pct = (step - warmup_steps) / (max_steps - warmup_steps)
      pct *= max_decay_factor
      values.append(lr - (pct * lr))

  boundaries = list(range(0, max_steps - 1))
  assert len(values) - 1 == len(boundaries)

  return PiecewiseConstantDecay(values=values,
                                boundaries=boundaries)


def main(argv):
  plan = load_plan('config/cnn_tune_lr.toml')
  tplan = plan.train
  lr = create_poly_schedule(tplan)
  for i in range(tplan.epochs * tplan.steps_per_epoch):
    print(i, lr(i))
  sys.exit(0)

  lr = create_warm(tplan)
  for e in range(tplan.epochs):
    print(e, lr(e))
  sys.exit(0)
  lrs = create_warm_linear_scheduler(tplan)
  func = lrs.schedule
  for e in range(tplan.epochs):
    print(e, func(e, None))


if __name__ == '__main__':
  app.run(main)
