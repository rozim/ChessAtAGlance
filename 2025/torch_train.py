from collections import Counter
import code
import sys
import time
import pprint
import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from absl import app
from absl import flags
from absl import logging

from torchinfo  import summary

from torch_model import MySimpleModel
from torch_data import MyDataset

import torchdata

flags.DEFINE_string('train',
                    'data/mega2600_shuffled_train.json',
                    'Output - train')
flags.DEFINE_string('test',
                    'data/mega2600_shuffled_test.json',
                    'Output - test')
flags.DEFINE_string('device', '', 'cpu/gpu/mps. If not set then tries for best avail.')

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


def run_eval(model, device, loss_fn, dl, limit):
  t1 = time.time()
  examples, total_loss, correct, correct_tot = 0, 0.0, 0, 0
  with torch.no_grad():
    for batch, entry in enumerate(dl):
      x = entry['board_1024']
      y = entry['move_1968']
      x, y = x.to(device), y.to(device)
      pred = model(x)
      loss = loss_fn(pred, y)
      total_loss += loss.item()
      nc = (pred.argmax(1) == y).type(torch.float).sum().item()
      correct += nc
      correct_tot += len(x)
      examples += len(x)
      if examples >= limit:
        break

  dt = time.time() - t1
  print(f'eval correct={correct:,} correct_tot={correct_tot:,} ratio={100.0 * correct / correct_tot:>6.3f}')
  print(f'loss {total_loss / batch:.1f}')
  print(f'time: {dt:.1f}s')


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

  model = MySimpleModel(n_channels=5,
                        n_blocks=3,
                        n_choke=7)
  model = model.to(device)

  d1 = MyDataset(FLAGS.train)
  d2 = ShuffleDataset(d1, buffer_size=1024)

  dl_train = torch.utils.data.DataLoader(d2, batch_size=64)


  loss_fn = nn.CrossEntropyLoss()
  print(loss_fn)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
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

  dt = time.time() - t1
  print(f'train time: {dt:.1f}s')
  model.eval()
  print()
  print('Eval: test')
  run_eval(model, device, loss_fn, torch.utils.data.DataLoader(MyDataset(FLAGS.test), batch_size=64), limit=10240)
  print()
  print('Eval: train')
  run_eval(model, device, loss_fn, torch.utils.data.DataLoader(MyDataset(FLAGS.train), batch_size=64), limit=10240)
  print()



if __name__ == '__main__':
  app.run(main)
