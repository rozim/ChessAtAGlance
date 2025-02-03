# Based on https://github.com/zach1502/BetaChess/tree/main
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader



NUM_RESIDUAL_BLOCKS = 4
BATCH_SIZE = 32
PATIENCE = 128
PRINT_INTERVAL = 32
LEARNING_RATE = 0.03
NUM_EPOCHS = 2048


class BoardData(Dataset):
  def __init__(self, dataset):  # dataset = np.array of (s, p, v)
    self.X = dataset[:, 0]
    self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx].transpose(2, 0, 1), self.y_p[idx], self.y_v[idx]


class ConvBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.action_size = 8 * 8 * 73
    self.conv1 = nn.Conv2d(20, 256, 3, stride=1, padding='same')
    self.bn1 = nn.BatchNorm2d(256)

  def forward(self, s):
    ##### old use of view?
    ##### s = s.view(-1, 20, 8, 8)  # batch_size x channels x board_x x board_y
    s = F.relu(self.bn1(self.conv1(s)))
    return s


class ResBlock(nn.Module):
  def __init__(self, inplanes=256, planes=256, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                           padding='same', bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding='same', bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = F.relu(self.bn1(out))
    out = self.conv2(out)
    out = self.bn2(out)
    out += residual
    out = F.relu(out)
    return out


class OutBlock(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(256, 1, kernel_size=1)  # value head
    self.bn = nn.BatchNorm2d(1)
    self.fc1 = nn.Linear(8 * 8, 64)
    self.fc2 = nn.Linear(64, 1)

    self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # policy head
    self.bn1 = nn.BatchNorm2d(128)
    self.logsoftmax = nn.LogSoftmax(dim=1)
    self.fc = nn.Linear(8 * 8 * 128, 8 * 8 * 73)

  def forward(self, s):
    v = F.relu(self.bn(self.conv(s)))  # value head
    v = v.view(-1, 8 * 8)  # batch_size X channel X height X width
    v = F.relu(self.fc1(v))
    v = torch.tanh(self.fc2(v))

    p = F.relu(self.bn1(self.conv1(s)))  # policy head
    p = p.view(-1, 8 * 8 * 128)
    p = self.fc(p)
    p = self.logsoftmax(p).exp()
    return p, v


class AzModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = ConvBlock()
    self.res_blocks = nn.ModuleList([ResBlock() for _ in range(NUM_RESIDUAL_BLOCKS)])
    self.outblock = OutBlock()

  def forward(self, s):
    s = self.conv(s)
    for block in self.res_blocks:
      s = block(s)
    s = self.outblock(s)
    return s # policy, value


# class ChessLoss(nn.Module):
#   def __init__(self):
#     super().__init__()

#   def forward(self, y_value, value, y_policy, policy):
#     value_error = (value - y_value) ** 2
#     policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()), 1)
#     total_error = (value_error.view(-1).float() + policy_error).mean()
#     return total_error
