from absl.testing import absltest

from encode import *


class EncodeTest(absltest.TestCase):
  def test_encode_transformer(self):
    fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
    before = chess.Board(fen)
    encoded = encode_transformer_board_wtm(before)
    after = decode_transformer_board_wtm(encoded)
    self.assertEqual(before.fen(), after.fen())

  def test_encode_transformer_orig(self):
    before = chess.Board()
    before.clear_board()
    before = chess.Board()
    encoded = encode_transformer_board_wtm(before)
    after = decode_transformer_board_wtm(encoded)
    self.assertEqual(before.fen(), after.fen())

  def test_encode_transformer_castle_stress1(self):
    fen = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1    '
    before = chess.Board(fen)
    encoded = encode_transformer_board_wtm(before)
    after = decode_transformer_board_wtm(encoded)
    self.assertEqual(before.fen(), after.fen())

  def test_encode_transformer_castle_stress2(self):
    fen = 'r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1'
    before = chess.Board(fen)
    encoded = encode_transformer_board_wtm(before)
    after = decode_transformer_board_wtm(encoded)
    self.assertEqual(before.fen(), after.fen())

  def test_encode_transformer_ep(self):
    fen = 'rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 1'
    print("FEN: ", fen)
    before = chess.Board(fen)
    print("BEFORE: ", before.fen())
    self.assertEqual(before.fen(), fen)
    encoded = encode_transformer_board_wtm(before)

    after = decode_transformer_board_wtm(encoded)
    print("AFTER: ", after.fen())
    self.assertEqual(before.fen(), after.fen())


if __name__ == '__main__':
  absltest.main()
