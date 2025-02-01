import chess
import torch
from absl.testing import absltest

from az_encode import *
from chess import *
import numpy

CRAZY = 'r3kb1r/pppq1pPp/2n1bn2/4p3/4P3/7N/pPP1BP1P/RNBQK2R w KQkq - 0 1'

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
    for co in [WHITE, BLACK]:
      b = chess.Board(CRAZY)
      b.turn = co
      already1 = set()
      already2 = set()

      for m in b.legal_moves:
        a = encode_action(m, b)
        assert 0 <= a < ACTIONS
        assert a not in already1
        already1.add(a)
        assert type(a) == np.int64
        m2 = decode_action(a, b)
        assert m2 not in already2
        already2.add(m2)
        assert m == m2
        assert type(m2) == chess.Move


  def test_crazy_board(self):
    b = chess.Board(CRAZY)
    enc = encode_board(b)
    b2 = decode_board(enc)
    assert b == b2
    assert b.fen() == b2.fen()
    assert enc.shape == (8, 8, 20)
    assert type(enc) == np.ndarray
    assert enc.dtype == np.int64



if __name__ == '__main__':
  absltest.main()
