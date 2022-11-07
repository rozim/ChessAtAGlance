import numpy as np

import chess
from chess import WHITE, BLACK
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

from absl import app
from absl import flags

CNN_SHAPE = (12, 64)
CNN_SHAPE_2D = (12, 8, 8)
CNN_FLAT_SHAPE = 12 * 64

TRANSFORMER_SHAPE = (64 + 1 + 4)

# P2I {
#   PAWN: 0,
#   KNIGHT: 1,
#   BISHOP: 2,
#   ROOK: 3,
#   QUEEN: 4,
#   KING: 5}

CO_P2I = [
  { # black
    PAWN: 0,
    KNIGHT: 1,
    BISHOP: 2,
    ROOK: 3,
    QUEEN: 4,
    KING: 5},
  { # white
    PAWN: 6,
    KNIGHT: 7,
    BISHOP: 8,
    ROOK: 9,
    QUEEN: 10,
    KING: 11}
]

TRANSFORMER_CO_P2I = [
  { # black
    PAWN: 1,
    KNIGHT: 2,
    BISHOP: 3,
    ROOK: 4,
    QUEEN: 5,
    KING: 6},
  { # white
    PAWN: 7,
    KNIGHT: 8,
    BISHOP: 9,
    ROOK: 10,
    QUEEN: 11,
    KING: 12}
]
TRANSFORMER_VOCABULARY = (1 + 12)

# I2P = {
#   0: PAWN,
#   1: KNIGHT,
#   2: BISHOP,
#   3: ROOK,
#   4: QUEEN,
#   5: KING
# }

CO_I2P = {
  0: (WHITE, PAWN),
  1: (WHITE, KNIGHT),
  2: (WHITE, BISHOP),
  3: (WHITE, ROOK),
  4: (WHITE, QUEEN),
  5: (WHITE, KING),

  6: (BLACK, PAWN),
  7: (BLACK, KNIGHT),
  8: (BLACK, BISHOP),
  9: (BLACK, ROOK),
  10: (BLACK, QUEEN),
  11: (BLACK, KING),
}


def encode_cnn_board(board):
  ar = np.zeros(CNN_SHAPE)
  for piece in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
    for color in [WHITE, BLACK]:
      for sq in board.pieces(piece, color):
        index = CO_P2I[color][piece]
        #print(piece, color, sq, index)
        assert ar[index][sq] == 0.0
        ar[index][sq] = 1.0
  return ar

def encode_transformer_board(board):
  ar = np.zeros(TRANSFORMER_SHAPE)
  for piece in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
    for color in [WHITE, BLACK]:
      for sq in board.pieces(piece, color):
        assert ar[sq] == 0.0
        print('set', sq, TRANSFORMER_CO_P2I[color][piece])
        ar[sq] = TRANSFORMER_CO_P2I[color][piece]

  ar[64] = 1.0 if board.turn else 0.0
  ar[65] = 1.0 if board.has_kingside_castling_rights(WHITE) else 0.0
  ar[66] = 1.0 if board.has_queenside_castling_rights(WHITE) else 0.0
  ar[67] = 1.0 if board.has_kingside_castling_rights(BLACK) else 0.0
  ar[68] = 1.0 if board.has_queenside_castling_rights(BLACK) else 0.0
  return ar



def main(argv):
  fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
  board = chess.Board(fen)

  for p in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
    for co in [WHITE, BLACK]:
      print('p=', p, 'co=', co, list(board.pieces(p, co)))
  print()
  print(encode_cnn_board(board))
  print()
  print(encode_transformer_board(board))
  print()
  print(board)


if __name__ == "__main__":
  app.run(main)
