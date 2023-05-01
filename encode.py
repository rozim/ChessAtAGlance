#
# CNN Planes:
#
#   WP, WN, WB, WR, WQ, WK
#   BP, BN, BB, BR, BQ, BK
#   Castle: WQ, BQ, WK, BK
#
#
#
import numpy as np

import chess
from chess import WHITE, BLACK
from chess import PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING

from absl import app
from absl import flags

import tensorflow as tf

from encode_move import MOVE_TO_INDEX

NUM_CLASSES = len(MOVE_TO_INDEX)

CNN_PLANES = (12 + 4) # 12 pieces, 4 castle
CNN_SHAPE_2D = (CNN_PLANES, 64)
CNN_SHAPE_3D = (CNN_PLANES, 8, 8)
CNN_FLAT_SHAPE = CNN_PLANES * 64
CNN_ONES_PLANE = np.ones(64)

# TRANSFORMER_SIZE = (64 + 1 + 4)
# TRANSFORMER_VOCABULARY = (1 + 12)

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
CASTLE_WQ, CASTLE_BQ, CASTLE_WK, CASTLE_BK = (12, 13, 14, 15)
assert CASTLE_BK == (CNN_PLANES - 1), (CASTLE_BK, CNN_PLANES)

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


CNN_FEATURES = {
  'board': tf.io.FixedLenFeature(CNN_SHAPE_3D, tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64)
}

def encode_cnn_board_move_wtm(board, move):
  if board.turn == BLACK:
    # As move was already played.. (confusing)
    move = chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)
  return (encode_cnn_board_wtm(board),
          MOVE_TO_INDEX[move.uci()])

def encode_cnn_board_wtm(board):
  """Encode in WTM mode.
  No 'turn' is in the output.
  """
  if board.turn == BLACK:
    board = board.mirror()
  ar = np.zeros(CNN_SHAPE_2D)
  for piece in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
    for color in [WHITE, BLACK]:
      for sq in board.pieces(piece, color):
        index = CO_P2I[color][piece]
        #print(piece, color, sq, index)
        assert ar[index][sq] == 0.0
        ar[index][sq] = 1.0

  if board.has_queenside_castling_rights(WHITE):
    #print('castle/1')
    ar[CASTLE_WQ] = CNN_ONES_PLANE

  if board.has_queenside_castling_rights(BLACK):
    #print('castle/2')
    ar[CASTLE_BQ] = CNN_ONES_PLANE

  if board.has_kingside_castling_rights(WHITE):
    #print('castle/3')
    ar[CASTLE_WK] = CNN_ONES_PLANE

  if board.has_kingside_castling_rights(BLACK):
    #print('castle/4')
    ar[CASTLE_BK] = CNN_ONES_PLANE

  return ar

def encode_transformer_board(board):
  assert False, "needs to be revisited circa 2023-04-29"
  ar = np.zeros(TRANSFORMER_SIZE)
  for piece in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
    for color in [WHITE, BLACK]:
      for sq in board.pieces(piece, color):
        assert ar[sq] == 0.0
        # print('set', sq, TRANSFORMER_CO_P2I[color][piece])
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
  #board = chess.Board()

  for p in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
    for co in [WHITE, BLACK]:
      print('p=', p, 'co=', co, list(board.pieces(p, co)))
  print()
  encoded = encode_cnn_board_wtm(board)
  print(encoded)
  print()
  for i in range(12):
    print(i, np.nonzero(encoded[i]))
  for i in range(12, 16):
    print('castle', i, np.all(encoded[i] == CNN_ONES_PLANE))
  #print(encode_transformer_board(board))

  print()
  print(board)
  print()
  print('-----')
  e1, e2 = encode_cnn_board_move_wtm(board, chess.Move.from_uci('h2h3'))
  print('move: ', e2)


if __name__ == "__main__":
  app.run(main)
