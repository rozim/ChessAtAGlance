from absl import app
from absl import flags
from absl import logging


from chess import *


def main(argv):

  b = Board()
  all = set()
  for sq in range(64):
    for co in [WHITE, BLACK]:
      for ptype in [QUEEN, KNIGHT, PAWN]:
        b.clear_board()
        b.turn = co

        if ptype == PAWN:
          for sq2 in range(64):
            if abs(sq2 - sq) != 8: # not in front of
              b.set_piece_at(sq2, Piece(QUEEN, not co))
        b.set_piece_at(sq, Piece(ptype, co))
        all.update([m.uci() for m in b.legal_moves])

  print('\n'.join(sorted(all)))


if __name__ == '__main__':
  app.run(main)
