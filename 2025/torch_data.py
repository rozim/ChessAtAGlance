import pprint

import jsonlines
import torch
from absl import app, flags

FLAGS = flags.FLAGS



class MyDataset(torch.utils.data.IterableDataset):
  def __init__(self, fn: str):
    # self.f = gzip.GzipFile(fn, 'rb')
    # self.f = file(fn, 'r')
    self.reader = jsonlines.open(fn)

  def __iter__(self):
    for j_obj in self.reader:
      yield ({'board_1024': torch.tensor(j_obj['board_1024']).float(),
              'move_1968': torch.tensor(j_obj['move_1968'])
              })



def main(argv):
  #FN_TRAIN = '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled-train.jsonl.gz'
  #FN_TEST = '/Users/dave/Projects/ChessAutoencoder/chess_autoencoder/sparse/data/mega-2400-shuffled-test.jsonl.gz'
  #r = jsonlines.Reader('data/twic1174.jsonl')
  # r = jsonlines.open('data/twic1174.jsonl')
  # for foo in r:
  #   print(foo)
  #   break
  # sys.exit(1)
  ds = MyDataset('data/twic1174.jsonl')

  dl = torch.utils.data.DataLoader(ds)

  for ent in dl:
    pprint.pprint(ent)
    t = ent['board_1024']

    print(t.shape)
    assert t.shape == (1, 1024), t.shape
    t = ent['move_1968']
    print(t.shape)
    assert t.shape == (1,), t.shape
    break




if __name__ == '__main__':
  app.run(main)
