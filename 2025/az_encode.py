from absl import app

# Based on https://github.com/zach1502/BetaChess/tree/main
import numpy as np
import chess

import numpy.typing as npt

KNIGHT_MOVEMENT_DXDY_TO_IDX = {(2, -1): 0, (2, 1): 1, (1, -2): 2, (-1, -2): 3, (-2, 1): 4, (-2, -1): 5, (-1, 2): 6, (1, 2): 7}
KNIGHT_MOVEMENT_IDX_TO_DXDY = [(2, -1), (2, 1), (1, -2), (-1, -2), (-2, 1), (-2, -1), (-1, 2), (1, 2)]

ACTIONS = (8 * 8 * 73)
BOARD_SHAPE = (8, 8, 20)

_ENCODER_DICT = {
  "R": 0, "N": 1, "B": 2, "Q": 3, "K": 4, "P": 5,
  "r": 6, "n": 7, "b": 8, "q": 9, "k": 10, "p": 11}

_DECODER_DICT = {0: "R", 1: "N", 2: "B", 3: "Q", 4: "K", 5: "P", 6: "r", 7: "n", 8: "b", 9: "q", 10: "k", 11: "p"}

def encode_board(board: chess.Board) -> npt.NDArray[np.int64]:
    encoded = np.zeros([8, 8, 20], dtype=int)

    for square, piece in board.piece_map().items():
        i, j = chess.square_rank(square), chess.square_file(square)
        encoded[i, j, _ENCODER_DICT[piece.symbol()]] = 1

    encoded[:, :, 12] = board.turn == chess.WHITE

    encoded[:, :, 13] = board.has_queenside_castling_rights(chess.WHITE)
    encoded[:, :, 14] = board.has_kingside_castling_rights(chess.WHITE)
    encoded[:, :, 15] = board.has_queenside_castling_rights(chess.BLACK)
    encoded[:, :, 16] = board.has_kingside_castling_rights(chess.BLACK)

    encoded[:, :, 17] = board.fullmove_number
    encoded[:, :, 18] = board.halfmove_clock

    if board.ep_square is not None:
        i, j = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        encoded[i, j, 19] = 1

    return encoded

def decode_board(encoded: np.ndarray) -> chess.Board:
    board = chess.Board.empty()

    piece_positions = np.argwhere(encoded[:, :, :12] == 1)
    for i, j, k in piece_positions:
        board.set_piece_at(chess.square(j, i), chess.Piece.from_symbol(_DECODER_DICT[k]))

    board.turn = chess.WHITE if encoded[0, 0, 12] == 1 else chess.BLACK

    if encoded[0, 0, 13]:
        board.castling_rights |= chess.BB_A1
    if encoded[0, 0, 14]:
        board.castling_rights |= chess.BB_H1
    if encoded[0, 0, 15]:
        board.castling_rights |= chess.BB_A8
    if encoded[0, 0, 16]:
        board.castling_rights |= chess.BB_H8

    board.fullmove_number = int(encoded[0, 0, 17])
    board.halfmove_clock = int(encoded[0, 0, 18])

    for i in [2, 5]:
        for j in range(8):
            if encoded[i, j, 19]:
                ep_square = chess.square(j, i)
                board.ep_square = ep_square
    return board


def encode_action(move, board) -> npt.NDArray[np.int64]:
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    from_row, from_col = divmod(from_square, 8)
    to_row, to_col = divmod(to_square, 8)

    dx, dy = to_row - from_row, to_col - from_col

    encoded = np.zeros([8, 8, 73]).astype(int)

    piece = board.piece_type_at(from_square)

    if piece in [chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KING] or (
    piece == chess.PAWN and promotion in [None, chess.QUEEN]):
        if dx != 0 and dy == 0:
            idx = 7 + dx if dx < 0 else 6 + dx
        elif dx == 0 and dy != 0:
            idx = 21 + dy if dy < 0 else 20 + dy
        elif dx == dy:
            idx = 35 + dx if dx < 0 else 34 + dx
        elif dx == -dy:
            idx = 49 + dx if dx < 0 else 48 + dx
    elif piece == chess.KNIGHT:
        idx = 56 + KNIGHT_MOVEMENT_DXDY_TO_IDX[(dx, dy)]
    elif piece == chess.PAWN and to_row in [0, 7] and promotion is not None:
        underpromotion_dict = {chess.ROOK: 0, chess.KNIGHT: 1, chess.BISHOP: 2}
        if abs(dx) == 1 and dy == 0:
            idx = 64 + underpromotion_dict[promotion]
        elif abs(dx) == 1 and dy == -1:
            idx = 67 + underpromotion_dict[promotion]
        elif abs(dx) == 1 and dy == 1:
            idx = 70 + underpromotion_dict[promotion]

    # TBD: replace with 1 liner (8 * from_row + from_col) * ... + idx?
    encoded[from_row, from_col, idx] = 1
    encoded = encoded.reshape(-1)
    encoded = np.argmax(encoded)

    return encoded


def decode_action(encoded: int, board: chess.Board) -> chess.Move:
    encoded_array = np.zeros(8 * 8 * 73, dtype=int)
    encoded_array[encoded] = 1
    encoded_array = encoded_array.reshape(8, 8, 73)

    from_row, from_col, idx = np.where(encoded_array == 1)
    from_row, from_col, idx = from_row[0], from_col[0], idx[0]

    from_square = (from_row << 3) + from_col
    promotion = None

    def calculate_to_square(dx, dy):
        to_row = from_row + dx
        to_col = from_col + dy
        return (to_row << 3) + to_col

    piece = board.piece_type_at(from_square)

    if idx < 7:
        dy = 0
        dx = idx - 7
    elif idx < 14:
        dy = 0
        dx = idx - 6
    elif idx < 21:
        dx = 0
        dy = idx - 21
    elif idx < 28:
        dx = 0
        dy = idx - 20
    elif idx < 35:
        dx = dy = idx - 35
    elif idx < 42:
        dx = dy = idx - 34
    elif idx < 49:
        dx = idx - 49
        dy = -dx
    elif idx < 56:
        dx = idx - 48
        dy = -dx
    elif idx < 64:
        dx, dy = KNIGHT_MOVEMENT_IDX_TO_DXDY[idx - 56]
    else:
        if (board.turn == chess.WHITE):
            dx = 1
            dy = [0, -1, 1][((idx-64)//3) % 3]
        else:
            dx = -1
            dy = [0, -1, 1][((idx-64)//3) % 3]

        promotion = {0: chess.ROOK, 1: chess.KNIGHT, 2: chess.BISHOP}[(idx-64) % 3]


    to_row = from_row + dx
    if piece == chess.PAWN and promotion is None and to_row in [0, 7]:
        promotion = chess.QUEEN

    to_square = calculate_to_square(dx, dy)
    return chess.Move(from_square, to_square, promotion=promotion)

def main(_):
  crazy = 'r3kb1r/pppq1pPp/2n1bn2/4p3/4P3/7N/pPP1BP1P/RNBQK2R w KQkq - 0 1'
  b = chess.Board()
  for m in b.legal_moves:
    a = encode_action(m, b)
    print(m, a, a.shape)
  print()
  print('crazy')
  b = chess.Board(crazy)
  b.turn = chess.WHITE
  for m in b.legal_moves:
    a = encode_action(m, b)
    m2 = decode_action(a, b)
    print(a, m, m2, b.san(m))
  print()
  print('crazy/2')
  b = chess.Board(crazy)
  b.turn = chess.BLACK
  for m in b.legal_moves:
    a = encode_action(m, b)
    m2 = decode_action(a, b)
    print(a, m, m2, b.san(m))
  print()


if __name__ == '__main__':
  app.run(main)
