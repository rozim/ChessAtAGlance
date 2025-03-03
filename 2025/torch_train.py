import random
import sys
import time
from collections import Counter

import toml
import torch
from absl import app, flags
from torch import nn

from objdict import objdict
from torch_data import MyDataset
from torch_eval import run_eval
from torch_model import MySimpleModel

flags.DEFINE_string('train',
                    'data/mega2600_shuffled_train.jsonl.gz',
                    'Output - train')
flags.DEFINE_string('test',
                    'data/mega2600_shuffled_test.jsonl.gz',
                    'Output - test')
flags.DEFINE_string('device', None, 'cpu/gpu/mps. If not set then tries for best avail.')
flags.DEFINE_string('plan', None, 'toml file')
flags.mark_flags_as_required(['plan'])

FLAGS = flags.FLAGS

# https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
class ShuffleDataset(torch.utils.data.IterableDataset):
  def __init__(self, dataset, buffer_size):
    super().__init__()
    self.dataset = dataset
    self.buffer_size = buffer_size

  def __iter__(self):
    shufbuf = []
    try:
      dataset_iter = iter(self.dataset)
      for i in range(self.buffer_size):
        shufbuf.append(next(dataset_iter))
    except:
      self.buffer_size = len(shufbuf)

    try:
      while True:
        try:
          item = next(dataset_iter)
          evict_idx = random.randint(0, self.buffer_size - 1)
          yield shufbuf[evict_idx]
          shufbuf[evict_idx] = item
        except StopIteration:
          break
      while len(shufbuf) > 0:
        yield shufbuf.pop()
    except GeneratorExit:
      pass




def load_plan(fn):
  return toml.load(fn, objdict)


def main(argv):
  if FLAGS.device:
    device = FLAGS.device
  else:
    device = (
      "cuda" if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available()
      else "cpu"
    )
  print('device: ', device)
  plan = load_plan(FLAGS.plan)
  tplan = plan.train
  mplan = plan.model
  dplan = plan.data

  toml.dump(plan, sys.stdout)

  model = MySimpleModel(mplan)
  model = model.to(device)

  d1 = MyDataset(FLAGS.train)
  d2 = ShuffleDataset(d1, buffer_size=1024)

  dl_train = torch.utils.data.DataLoader(d2, batch_size=dplan.batch_size)


  loss_fn = nn.CrossEntropyLoss()
  print(loss_fn)
  optimizer = torch.optim.AdamW(model.parameters(), lr=tplan.lr)
  print(optimizer)

  model.train()
  y_stats = Counter()
  correct, correct_tot = 0, 0
  total_examples = 0
  reports = 0
  freq = 100
  t1 = time.time()

  best_acc = 0.0
  best_loss = 999
  loss_stale = 0
  acc_stale = 0

  for batch, entry in enumerate(dl_train):
    x = entry['board_1024']
    total_examples += len(x)
    y = entry['move_1968']
    y_stats.update(y.numpy().tolist())
    x, y = x.to(device), y.to(device)

    pred = model(x)
    loss = loss_fn(pred, y)
    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct_tot += len(x)

    if optimizer:
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    if batch % freq == 0:
      torch.save(model.state_dict(), 'checkpoint.pt')
      #torch.jit.script(model).save('checkpoint.pt')

      dt = time.time() - t1
      loss = loss.item()
      xps = total_examples / dt
      acc = correct / correct_tot if correct_tot > 0 else 0.0
      if loss < best_loss:
        best_loss = loss
        loss_stale = 0
      else:
        loss_stale += 1
      if acc > best_acc:
        best_acc = acc
        acc_stale = 0
      else:
        acc_stale += 1
      print(f"{reports:6d}. {dt:5.1f}s, xps: {xps:6.1f}, loss: {loss:>6.2f} ({loss_stale:4d}), accuracy: {100.0 * correct / correct_tot:>6.2f} ({acc_stale:4d})")
      reports += 1
      #print('c: ', y_stats.most_common(10), y_stats.total())
      correct, correct_tot = 0, 0
      # if batch >= 1000: break
      if (tplan.loss_stale > 0 and loss_stale >= tplan.loss_stale) or (tplan.acc_stale > 0 and acc_stale >= tplan.acc_stale):
        print('# Stale')
        break

  torch.save(model.state_dict(), 'model.pt')
  dt = time.time() - t1
  print(f'train time: {dt:.1f}s')
  model.eval()
  print()
  print('Eval: test')
  run_eval(model, device, loss_fn, torch.utils.data.DataLoader(MyDataset(FLAGS.test), batch_size=dplan.batch_size), limit=10240)
  print()
  print('Eval: train')
  run_eval(model, device, loss_fn, torch.utils.data.DataLoader(MyDataset(FLAGS.train), batch_size=dplan.batch_size), limit=10240)
  print()



if __name__ == '__main__':
  app.run(main)
