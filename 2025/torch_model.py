import code
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from absl import app
from absl import flags
from absl import logging

from torchinfo  import summary


FLAGS = flags.FLAGS

IN_CHANNELS = 16

# https://github.com/geochri/AlphaZero_Chess/blob/master/src/alpha_net.py
# Slightly modified.
#
class ResBlock(nn.Module):
  def __init__(self,
               n_channels: int):
    super(ResBlock, self).__init__()
    self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3,
                           padding='same', bias=False)
    self.bn1 = nn.BatchNorm2d(n_channels)
    self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3,
                           padding='same', bias=False)
    self.bn2 = nn.BatchNorm2d(n_channels)

  def forward(self, x):
    residual = x
    x = self.conv1(x)
    x = F.relu(self.bn1(x))
    x = self.conv2(x)
    x = self.bn2(x)
    x += residual
    x = F.relu(x)
    return x


class MySimpleModel(nn.Module):
  def __init__(self,
               n_blocks: int,
               n_channels: int,
               n_choke: int):
    super().__init__()

    self.n_blocks = n_blocks
    self.project = nn.Conv2d(in_channels=IN_CHANNELS,
                             out_channels=n_channels,
                             kernel_size=3,
                             padding='same',
                             bias=False)

    for block in range(n_blocks):
      setattr(self, f'res_{block}', ResBlock(n_channels=n_channels))

    self.flatten = nn.Flatten()
    self.logits = nn.LazyLinear(1968)


  def forward(self, x):
    x = x.view(-1, IN_CHANNELS, 8, 8) # should be [BS, 1024]
    x = self.project(x)

    for block in range(self.n_blocks):
      x = getattr(self, f'res_{block}')(x)

    x = self.flatten(x)
    x = self.logits(x)

    return x

def main(argv):
  device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
  )
  device = 'cpu'

  print('device: ', device)

  model = MySimpleModel(n_channels=5,
                        n_blocks=3,
                        n_choke=7)
  model = model.to(device)
  print('Model:')
  print(model)
  print()
  print('Summary:')
  summary(model, (1, 1024,), dtypes=(torch.float,), depth=99)
  print()
  y = model(torch.rand(1, 1024))
  print(y.shape)



if __name__ == '__main__':
  app.run(main)
