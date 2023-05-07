from absl import app
from plan import *
import functools
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.callbacks import LearningRateScheduler

def create_lr_schedule(tplan):
  """ Returns callback"""
  if tplan.lr_schedule == 'warm_linear':
    return (create_warm_linear_schedule(tplan), tplan.lr)
  assert tplan.lr_schedule == 'cosine'

  lr = CosineDecayRestarts(initial_learning_rate=tplan.lr,
                           first_decay_steps=tplan.first_decay_steps,
                           t_mul=1,
                           m_mul=1,
                           alpha=tplan.alpha)
  return lr


def create_warm_linear_schedule(tplan):
  def _create_warm_linear_schedule(epoch, ignore_lr):
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
  return LearningRateScheduler(_create_warm_linear_schedule)


def main(argv):
  plan = load_plan('config/cnn_tune_1.toml')
  tplan = plan.train
  lrs = create_warm_linear_schedule(tplan)
  func = lrs.schedule
  for e in range(tplan.epochs):
    print(e, func(e, None))


if __name__ == '__main__':
  app.run(main)
