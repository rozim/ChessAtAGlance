# Based on https://github.com/zach1502/BetaChess/tree/main
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import dataclasses



@dataclasses.dataclass
class AzModelConfig:
  num_blocks: int = 4
  num_channels: int = 256



class ConvBlock(nn.Module):
  def __init__(self, config: AzModelConfig):
    super().__init__()
    self.action_size = 8 * 8 * 73
    self.conv1 = nn.Conv2d(20, config.num_channels, 3, stride=1, padding='same')
    self.bn1 = nn.BatchNorm2d(config.num_channels)

  def forward(self, s):
    ##### old use of view?
    ##### s = s.view(-1, 20, 8, 8)  # batch_size x channels x board_x x board_y
    s = F.relu(self.bn1(self.conv1(s)))
    return s


class ResBlock(nn.Module):
  def __init__(self, config: AzModelConfig):
    super().__init__()
    inplanes = config.num_channels
    planes = config.num_channels
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                           padding='same', bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
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
    v = torch.tanh(self.fc2(v)) # range: (-1, 1)

    p = F.relu(self.bn1(self.conv1(s)))  # policy head
    p = p.view(-1, 8 * 8 * 128)
    p = self.fc(p)
    p = self.logsoftmax(p).exp()
    return p, v


class AzModel(nn.Module):
  def __init__(self, config: AzModelConfig):
    super().__init__()
    self.conv = ConvBlock(config)
    self.res_blocks = nn.ModuleList([ResBlock(config) for _ in range(config.num_blocks)])
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
