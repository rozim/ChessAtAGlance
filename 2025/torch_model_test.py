import pprint

import torch
from absl.testing import absltest

from objdict import objdict
from torch_model import MySimpleModel


class TorchModelTest(absltest.TestCase):
  def test_model_output(self):
    model = MySimpleModel(objdict(n_channels=5,
                                  n_blocks=3,
                                  n_choke=7))
    y = model(torch.rand(1, 1024))
    assert y.dtype == torch.float32, y.dtype
    assert y.shape == (1, 1968), y.shape

    model = MySimpleModel(objdict(n_channels=5,
                                  n_blocks=0,
                                  n_choke=7))
    y = model(torch.rand(1, 1024))
    assert y.dtype == torch.float32, y.dtype
    assert y.shape == (1, 1968), y.shape


if __name__ == '__main__':
  absltest.main()
