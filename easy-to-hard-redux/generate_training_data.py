import warnings
import sys, os
import tensorflow as tf
import numpy as np
import pandas as pd

from absl import app
from absl import flags
from absl import logging
import time
import chess

FEATURES = {
  'puzzle': tf.io.FixedLenFeature((12 * 8 * 8,), tf.float32),
  'solution': tf.io.FixedLenFeature((64,), tf.float32),
}

# from easy-to-hard-data/make_chess.py
def get_board_tensor(board_str, black_moves):
  """ function to move from FEN representation to 12x8x8 tensor
    Note: The rows and cols in the tensor correspond to ranks and files
          in the same order, i.e. first row is Rank 1, first col is File A.
          Also, the color to move next occupies the first 6 channels."""

  p_to_int = {"k": 1, "q": 2, "b": 3, "n": 4, "r": 5, "p": 6,
                "K": 7, "Q": 8, "B": 9, "N": 10, "R": 11, "P": 12}
  new_board = np.zeros((8, 8))
  rank = 7
  file = 0
  for p in board_str:
    if p == "/":
      rank -= 1
      file = 0
    elif not p.isdigit():
      new_board[rank, file] = (p_to_int[p])
      file += 1
    else:
      new_board[rank, file:file+int(p)] = 0
      file += int(p)

  board_tensor = np.zeros((12, 8, 8))
  pieces = "kqbnrpKQBNRP" if black_moves else "KQBNRPkqbnrp"
  for p_i, p in enumerate(pieces):
    board_tensor[p_i] = (new_board == p_to_int[p])

  return board_tensor


def get_moves_tensor(move):
  file_to_num = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
  origin_file = file_to_num[move[0]]
  origin_rank = int(move[1]) - 1
  dest_file = file_to_num[move[2]]
  dest_rank = int(move[3]) - 1
  move = np.zeros((8, 8))
  move[origin_rank, origin_file] = 1
  move[dest_rank, dest_file] = 1
  return move


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def _extract(blob):
  t = tf.io.parse_example(blob, features=FEATURES)
  return ((t['puzzle'],),
          t['solution'])



def sq2i(sq):
  # e.g. h2
  row = ord(sq[0]) - ord('a')
  col = ord(sq[1]) - ord('1')
  return row * 8 + col

def move2i(m):
  return (sq2i(m[0:2]),
          sq2i(m[2:4]))


def main(argv):
  # print(sq2i('a1'))
  # print(sq2i('h1'))
  # print(sq2i('h8'))
  # print(move2i('a1h1'))

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  logging.set_verbosity('error')
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
  warnings.filterwarnings('ignore', category=Warning)

  if True:
    ds = tf.data.TFRecordDataset(['easy-v3.rio'], 'ZLIB', num_parallel_reads=1)
    ds = ds.map(_extract)
    ds = ds.batch(1)
    for ent in ds:
      print(ent)
      break
    sys.exit(0)

  df = pd.read_csv('lichess_db_puzzle.csv')
  t0 = time.time()
  mod = 1024
  opts = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    output_buffer_size=(4 * 1024 * 1024))

  df = df.sort_values('Rating')
  with tf.io.TFRecordWriter('easy-v3.rio', opts) as rio:

    index = -1
    for db_index, row in df.iterrows():
      index += 1
      rating = row['Rating']

      mistake = chess.Move.from_uci(row['Moves'].split(' ')[0])
      refutation = row['Moves'].split(' ')[1]
      board = chess.Board(row['FEN'])
      board.push(mistake)
      new_fen = board.fen()

      black_moves = {'w': 0, 'b': 1}[new_fen.split(' ')[1]]
      board_str = new_fen.split(' ')[0]
      puzzle_tensor = get_board_tensor(board_str, black_moves)
      moves_tensor = get_moves_tensor(refutation) # The refutation

      feature = {
        'puzzle': tf.train.Feature(float_list=tf.train.FloatList(value=puzzle_tensor.flatten())),
        'solution': tf.train.Feature(float_list=tf.train.FloatList(value=moves_tensor.flatten())),

        'refutation_uci':  _bytes_feature(refutation.encode('utf-8')),
        'new_fen':  _bytes_feature(new_fen.encode('utf-8')),
        'rating':  _int64_feature(rating),
      }

      pb = tf.train.Example(features=tf.train.Features(feature=feature))
      rio.write(pb.SerializeToString())

      if index % mod == 0:
        dt = int(time.time() - t0)
        print(index, dt, int(index/(dt+1)), int((100 * index) / 1857575), '%')
        mod *= 2


if __name__ == '__main__':
  app.run(main)
