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
from torch_eval import run_eval

FLAGS = flags.FLAGS
flags.DEFINE_string('device', None, 'cpu/gpu/mps. If not set then tries for best avail.')
flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_string('model_file', None, 'pt file')
flags.DEFINE_string('fen', None, 'FEN')

flags.mark_flags_as_required([ 'model_file', 'plan', 'fen'])

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
