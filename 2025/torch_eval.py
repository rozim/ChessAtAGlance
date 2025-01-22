import os
import time
from typing import Callable

import chess
import numpy as np
import toml
import torch
from absl import app, flags
from torch import nn

from encode import encode_cnn_board_wtm
from encode_move import MOVE_TO_INDEX
from objdict import objdict
from torch_model import MySimpleModel



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
