from absl.testing import absltest

from az_model import AzModel, AzModelConfig
from az_encode import *
import chess
import numpy as np
import torch


class AzModelTest(absltest.TestCase):
  def test_print(self):
    config = AzModelConfig()
    print(config)
    m = AzModel(config)
    print(m)

  def test_config(self):
    config = AzModelConfig(num_channels=7, num_blocks=3)
    print(config)
    m = AzModel(config)
    print(m)

  def test_with_encoded_board(self):
    enc = encode_board(chess.Board())      #     8, 8, 20
    enc = enc.transpose(2, 0, 1)           #    20, 8, 8
    batch = np.expand_dims(enc, axis=0)    # 1, 20, 8, 8
    print('shape', batch.shape)
    batch = torch.tensor(batch).float()
    model = AzModel(AzModelConfig())
    policy, value = model(batch)

    self.assertEqual(policy.shape, (1, 4672))
    self.assertEqual(policy.dtype, torch.float32)

    self.assertEqual(value.shape, (1, 1))
    self.assertEqual(value.dtype, torch.float32)


if __name__ == '__main__':
  absltest.main( )
