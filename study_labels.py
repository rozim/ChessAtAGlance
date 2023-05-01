import glob
import sys

from collections import Counter

import chess
import chess.pgn

from encode import encode_cnn_board_move_wtm
from pychess_util import *
from tf_util import *


pgn_fns = '../ChessData/Twic/twic47?.pgn'

c_flip = Counter()
c_orig = Counter()
tot = 0
flip, nflip = 0, 0

for pgn_fn in glob.glob(pgn_fns):
  for game in gen_games(pgn_fn):
    for (move, board) in gen_moves(game):
      c_orig[move.uci()] += 1

      if board.turn == chess.BLACK:
        flip += 1
        move = chess.Move(chess.square_mirror(move.from_square),
                          chess.square_mirror(move.to_square),
                          move.promotion)
      else:
        nflip += 1
      tot += 1
      c_flip[move.uci()] += 1

print('tot: ', tot)
print('flip: ', flip)
print('nflip: ', nflip)
print()
print('mc/1 (flip)', c_flip.most_common(10))
print('mc/2 (orig)', c_orig.most_common(10))

print()
for k, v in c_flip.items():
  print(f'{k:8s} {v:8d} {v/tot*100:.1f}')
