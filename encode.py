#
# CNN Planes:
#
#   WP, WN, WB, WR, WQ, WK
#   BP, BN, BB, BR, BQ, BK
#   Castle: WQ, BQ, WK, BK
#
#
#
import sys

import numpy as np

import chess
from chess import WHITE, BLACK, Piece
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


# 64: squares
# 4: castle
# 1: ep square
TRANSFORMER_LENGTH = (64 + 4 + 1)
TRANSFORMER_SHAPE = (TRANSFORMER_LENGTH,)
TRANSFORMER_VOCABULARY = 38

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

# Unique tokens.
# reserve 0 for empty square
# 1..12 for pieces
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

TRANSFORMER_CO_I2P = {
  1: (BLACK, PAWN),
  2: (BLACK, KNIGHT),
  3: (BLACK, BISHOP),
  4: (BLACK, ROOK),
  5: (BLACK, QUEEN),
  6: (BLACK, KING),

  7: (WHITE, PAWN),
  8: (WHITE, KNIGHT),
  9: (WHITE, BISHOP),
  10: (WHITE, ROOK),
  11: (WHITE, QUEEN),
  12: (WHITE, KING),
}



TRANSFORMER_INDEX_WK = 64
TRANSFORMER_INDEX_WQ = 65
TRANSFORMER_INDEX_BK = 66
TRANSFORMER_INDEX_BQ = 67
TRANSFORMER_INDEX_EP = 68

# 13..20 for Castling
TRANSFORMER_CASTLE_WK = 13
TRANSFORMER_CASTLE_NO_WK = 14
TRANSFORMER_CASTLE_WQ = 15
TRANSFORMER_CASTLE_NO_WQ = 16
TRANSFORMER_CASTLE_BK = 17
TRANSFORMER_CASTLE_NO_BK = 18
TRANSFORMER_CASTLE_BQ = 19
TRANSFORMER_CASTLE_NO_BQ = 20

# 21..37 for EP squares
TRANSFORMER_EP = {
  chess.A3: 21,
  chess.B3: 22,
  chess.C3: 23,
  chess.D3: 24,
  chess.E3: 25,
  chess.F3: 26,
  chess.G3: 27,
  chess.H3: 28,

  chess.A6: 29,
  chess.B6: 30,
  chess.C6: 31,
  chess.D6: 32,
  chess.E6: 33,
  chess.F6: 34,
  chess.G6: 35,
  chess.H6: 36,

  None: 37

  }

TRANSFORMER_EP_DECODE = {
  v: k for k, v in TRANSFORMER_EP.items()
  }





CNN_FEATURES = {
  'board': tf.io.FixedLenFeature(CNN_SHAPE_3D, tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64)
}
CNN_FEATURES_FEN = {
  'board': tf.io.FixedLenFeature(CNN_SHAPE_3D, tf.float32),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'fen': tf.io.FixedLenFeature([], tf.string),
}

TRANSFORMER_FEATURES = {
  'board': tf.io.FixedLenFeature(TRANSFORMER_SHAPE, tf.int64),
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

def encode_transformer_board_move_wtm(board, move):
  if board.turn == BLACK:
    # As move was already played.. (confusing)
    move = chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      move.promotion)
  return (encode_transformer_board_wtm(board),
          MOVE_TO_INDEX[move.uci()])

def encode_transformer_board_wtm(board):
  ar = np.zeros(TRANSFORMER_LENGTH, dtype=np.int8)
  for piece in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING]:
    for color in [WHITE, BLACK]:
      for sq in board.pieces(piece, color):
        assert ar[sq] == 0.0
        # print('set', sq, TRANSFORMER_CO_P2I[color][piece])
        ar[sq] = TRANSFORMER_CO_P2I[color][piece]
        #print('SET ', 'sq', sq, 'enc', ar[sq], 'p', piece, 'co', color)

  ar[TRANSFORMER_INDEX_WK] = TRANSFORMER_CASTLE_WK if board.has_kingside_castling_rights(WHITE)  else TRANSFORMER_CASTLE_NO_WK
  ar[TRANSFORMER_INDEX_WQ] = TRANSFORMER_CASTLE_WQ if board.has_queenside_castling_rights(WHITE) else TRANSFORMER_CASTLE_NO_WQ
  ar[TRANSFORMER_INDEX_BK] = TRANSFORMER_CASTLE_BK if board.has_kingside_castling_rights(BLACK)  else TRANSFORMER_CASTLE_NO_BK
  ar[TRANSFORMER_INDEX_BQ] = TRANSFORMER_CASTLE_BQ if board.has_queenside_castling_rights(BLACK) else TRANSFORMER_CASTLE_NO_BQ
  ar[TRANSFORMER_INDEX_EP] = TRANSFORMER_EP[board.ep_square]

  return ar

def decode_transformer_board_wtm(ar):
  board = chess.Board()
  board.clear_board()
  for sq in range(64):
    encoded = ar[sq]
    if encoded != 0:
      co, p = TRANSFORMER_CO_I2P[encoded]
      board.set_piece_at(sq, Piece(p, co))
      #print('RESTORE ', 'sq', sq, 'enc', encoded, 'p', p, 'co', co, ' ', board.fen())

  castling_fen = ""
  if ar[TRANSFORMER_INDEX_WK] == TRANSFORMER_CASTLE_WK:
    castling_fen += "K"
  if ar[TRANSFORMER_INDEX_WQ] == TRANSFORMER_CASTLE_WQ:
    castling_fen += "Q"

  if ar[TRANSFORMER_INDEX_BK] == TRANSFORMER_CASTLE_BK:
    castling_fen += "k"
  if ar[TRANSFORMER_INDEX_BQ] == TRANSFORMER_CASTLE_BQ:
    castling_fen += "q"

  board.set_castling_fen(castling_fen)

  board.ep_square = TRANSFORMER_EP_DECODE[ar[TRANSFORMER_INDEX_EP]]

  return board




def main(argv):
  fen = 'r1bq1rk1/4bppp/p2p1n2/npp1p3/4P3/2P2N1P/PPBP1PP1/RNBQR1K1 w - -'
  board = chess.Board(fen)
  board = chess.Board()
  print('BEFORE: ', board.fen())
  #board = chess.Board()
  with np.printoptions(linewidth=999):
    encoded = encode_transformer_board_wtm(board)
    print(encoded)
  board = decode_transformer_board_wtm(encoded)
  print(board)
  print('AFTER: ', board.fen())
  sys.exit(0)

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

  print()
  print(board)
  print()
  print('-----')
  e1, e2 = encode_cnn_board_move_wtm(board, chess.Move.from_uci('h2h3'))
  print('move: ', e2)


if __name__ == "__main__":
  app.run(main)
