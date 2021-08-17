from absl import app
from plan import *
import functools
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

def create_lr_schedule(tplan):
  if tplan.lr_schedule == 'warm_linear':
    return functools.partial(create_warm_linear_schedule, tplan)
  assert tplan.lr_schedule == 'cosine'
  
  lr = CosineDecayRestarts(initial_learning_rate=tplan.lr,
                           first_decay_steps=tplan.first_decay_steps,
                           t_mul=1,
                           m_mul=1,
                           alpha=tplan.alpha)
  return lr

  
def create_warm_linear_schedule(tplan, epoch):
  lr, train_epochs, train_warmup = (tplan.lr,
                                    tplan.epochs,
                                    tplan.warmup)


  epoch = epoch + 1.0

  if epoch <= train_warmup:

    return lr * (epoch / train_warmup)

  else:
    pct = (epoch - train_warmup) / (train_epochs - train_warmup)
    return lr - (pct * lr)


def main(argv):
  plan = load_plan('v4.toml')
  tplan = plan.train
  schedule = functools.partial(create_warm_linear_schedule, tplan)
  for e in range(tplan.epochs):
    print(e, schedule(e))
    
if __name__ == '__main__':
  app.run(main)      
