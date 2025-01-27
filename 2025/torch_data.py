import pprint

import jsonlines
import gzip
import torch
import time
from absl import app, flags
from encode import encode_cnn_board_move_wtm
import chess
FLAGS = flags.FLAGS



class MyDataset(torch.utils.data.IterableDataset):
  def __init__(self, fn: str):
    # self.f = gzip.GzipFile(fn, 'rb')
    # self.f = file(fn, 'r')
    if fn.endswith('.gz'):
      # self.reader = jsonlines.open(gzip.open(fn, 'rt'))
      self.reader = jsonlines.Reader(gzip.open(fn, 'rt'))
    else:
      self.reader = jsonlines.open(fn)

  def __iter__(self):
    for j_obj in self.reader:
      try:
        yield ({'board_1024': torch.tensor(j_obj['board_1024']).float(),
                'move_1968': torch.tensor(j_obj['move_1968'])
                })
      except KeyError:
        sfen = j_obj['sfen']
        san = j_obj['move_san']
        board = chess.Board(sfen)
        move = board.parse_san(san)
        board_1024, move_1968 = encode_cnn_board_move_wtm(board, move)
        yield ({'board_1024': torch.tensor(board_1024.flatten()).float(),
                'move_1968': torch.tensor(move_1968)
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
  # ds = MyDataset('data/twic1174.jsonl')
  ds = MyDataset('data/twic1200-sf-gen1-shuffled.jsonl.gz')

  dl = torch.utils.data.DataLoader(ds)

  t1 = time.time()
  mod = 1
  for i, ent in enumerate(dl):
    if i == 0:
      pprint.pprint(ent)
      t = ent['board_1024']

      print(t.shape)
      assert t.shape == (1, 1024), t.shape
      t = ent['move_1968']
      print(t.shape)
      assert t.shape == (1,), t.shape
    if i % mod == 0:
      t2 = time.time()
      dt = t2 - t1
      rate = mod / dt
      print(f'{dt:.1f}s {i}: {rate}')
      t1 = t2
      mod *= 2





if __name__ == '__main__':
  app.run(main)
