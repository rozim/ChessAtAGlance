from collections import Counter
import os
import code
import sys
import time
import pprint
import random
import toml
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import chess

import numpy as np

from absl import app
from absl import flags
from absl import logging

from typing import Callable

from torchinfo  import summary
from torch_model import MySimpleModel
from torch_data import MyDataset
from objdict import objdict
from encode import encode_cnn_board_wtm
from encode_move import MOVE_TO_INDEX

FLAGS = flags.FLAGS
flags.DEFINE_string('device', None, 'cpu/gpu/mps. If not set then tries for best avail.')
flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_string('model_file', None, 'pt file')
flags.DEFINE_string('fen', None, 'FEN')

flags.mark_flags_as_required([ 'model_file', 'plan', 'fen'])

def run_eval(model: nn.Module,
             device: str,
             loss_fn: Callable,
             dl: torch.utils.data.DataLoader,
             limit: int):
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




def softmax(x):
    # Subtract the maximum value from x for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def main(argv):
  assert os.path.exists(FLAGS.model_file)
  assert os.path.exists(FLAGS.plan)
  assert FLAGS.model_file.endswith('pt')
  if FLAGS.device:
    device = FLAGS.device
  else:
    device = (
      "cuda" if torch.cuda.is_available()
      else "mps" if torch.backends.mps.is_available()
      else "cpu"
    )
  print('device: ', device)
  plan = toml.load(FLAGS.plan, objdict)
  # tplan = plan.train
  mplan = plan.model
  # dplan = plan.data

  model = MySimpleModel(mplan)
  model.load_state_dict(torch.load(FLAGS.model_file, weights_only=True))
  model.eval()
  model = model.to(device)

  board = chess.Board(FLAGS.fen)
  with torch.no_grad():
    x = encode_cnn_board_wtm(board)
    x = np.expand_dims(x, axis=0)
    x = torch.tensor(x).float()
    x = x.to(device)
    y = model(x)
    ar = []
    mml = list(board.legal_moves)
    for m in mml:
      yi = MOVE_TO_INDEX[m.uci()]
      logit = y[0][yi].float().item()
      ar.append(logit)
    sm = softmax(np.array(ar))
    for i, m in enumerate(mml):
      print(f'{i:2d} | {m.uci():6s} | {board.san(m):6s} | {100.0*sm[i]:.0f}')


  #ds = MyDataset(FLAGS.eval)
  #dl = torch.utils.data.DataLoader(ds, batch_size=64)

if __name__ == '__main__':
  app.run(main)
