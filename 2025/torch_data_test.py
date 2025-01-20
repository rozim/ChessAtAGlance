from absl.testing import absltest

import torch
from torch_data import MyDataset
import pprint

class TorchDataTest(absltest.TestCase):
  def test_data_read(self):
    ds = MyDataset('testdata/data10.jsonl')
    dl = torch.utils.data.DataLoader(ds)
    n = 0
    for j in dl:
      pprint.pprint(j)
      n += 1
      assert n < 20
      t = j['board_1024']
      assert t.shape == (1,1024,), t.shape
      t = j['move_1968']
      assert t.shape == (1,), t.shape
    assert n == 10, n

if __name__ == '__main__':
  absltest.main()
