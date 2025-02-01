import chess
import torch
from absl.testing import absltest

from az_encode import *
from chess import *


class AzEncodeTest(absltest.TestCase):
  def test_all_queen_moves_unique(self):
    white = set()

    for sq in range(64):
      b = Board()
      b.clear_board()
      b.turn = WHITE
      b.set_piece_at(sq, Piece(QUEEN, WHITE))
      for m in b.legal_moves:
        a = encode_action(m, b)
        assert a not in white, m
        white.add(a)
    assert len(white) == 1456, len(white)

    # Black queen should be already there.
    black = set()
    for sq in range(64):
      b = Board()
      b.clear_board()
      b.turn = BLACK
      b.set_piece_at(sq, Piece(QUEEN, BLACK))
      for m in b.legal_moves:
        a = encode_action(m, b)
        assert a in white
        assert a not in black
        black.add(a)
    assert len(black) == 1456, len(black)

  def test_crazy_position_both_operations(self):
    crazy = 'r3kb1r/pppq1pPp/2n1bn2/4p3/4P3/7N/pPP1BP1P/RNBQK2R w KQkq - 0 1'

    for co in [WHITE, BLACK]:
      b = chess.Board(crazy)
      b.turn = co
      already1 = set()
      already2 = set()

      for m in b.legal_moves:
        a = encode_action(m, b)
        assert a not in already1
        already1.add(a)
        m2 = decode_action(a, b)
        assert m2 not in already2
        already2.add(m2)
        assert m == m2







if __name__ == '__main__':
  absltest.main()



    #   for co in [WHITE, BLACK]:
    #     for ptype in [QUEEN, KNIGHT, PAWN]:
    #       b = Board()
    #       b.clear_board()
    #       b.turn = co

    #       if ptype == PAWN:
    #         for sq2 in range(64):
    #           if abs(sq2 - sq) != 8: # not in front of
    #             b.set_piece_at(sq2, Piece(QUEEN, not co))
    #       b.set_piece_at(sq, Piece(ptype, co))
    #       for m in b.legal_moves:
    #         a = encode_action(m, b)
    #         assert a not in all, m
    #         all.add(a)
    # assert len(all) == 3, len(all)  a
