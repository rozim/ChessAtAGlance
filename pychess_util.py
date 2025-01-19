import chess
import chess.pgn

def gen_games(fn):
  with open(fn, encoding="ISO-8859-1") as pgn:
    while True:
      g = chess.pgn.read_game(pgn)
      if g is None:
        return
      yield g

def gen_games_pct(fn):
  with open(fn, 'r', encoding='utf-8', errors='replace') as f:
    fsize = f.seek(0, 2) # eof
    f.seek(0, 0) # rewind
    while True:
      pos = f.seek(0, 1)
      g = chess.pgn.read_game(f)
      if g is None:
        return
      yield g, (pos / fsize)


def gen_moves(game):
  ##? assert False, "prob broken"
  board = game.board()
  for move in game.mainline_moves():
    yield move, board
    board.push(move)

# gen_moves returned a board that is a singleton, needs to be duped
def gen_moves_fixed(game):
  board = game.board()
  for ply, move in enumerate(game.mainline_moves()):
    yield move.uci(), board.san(move), ply, board.fen()
    board.push(move)

def gen_moves_fixed_v2(game): # ugh, hack/misdesign
  board = game.board()
  for ply, move in enumerate(game.mainline_moves()):
    yield move, board, ply # careful: board only temporarily alive
    board.push(move)

def simplify_fen(board):
  #rn2kbnr/ppq2pp1/4p3/2pp2Bp/2P4P/1Q6/P2NNPP1/3RK2R w Kkq - 2 13
  return ' '.join(board.fen().split(' ')[0:4])

def simplify_fen_string(fen):
  return ' '.join(fen.split(' ')[0:4])


def gen_fens(game):
  board = game.board()
  for ply, move in enumerate(game.mainline_moves()):
    board.push(move)
    yield board.fen() # FEN after move
