import chess
import chess.pgn
import chess.engine
import sys, os
import random
import time

board = chess.Board()

print('legal_moves: ', list(board.legal_moves))
print(list(board.legal_moves)[0])
print(type(list(board.legal_moves)[0]))

m = list(board.legal_moves)[0]
print('m', m)
print('uci', m.uci())
print('san', board.san(m))
#sys.exit(0)


board.push_san("e4")

print(board)
print('out:', board.outcome())
print(board.fen())

STOCKFISH = '/opt/homebrew/bin/stockfish'

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

board = chess.Board('r1bq1b1r/pppnpkpp/8/3n4/3P4/8/PPP2PPP/RNBQKB1R w KQ - 0 7')
print(board)
limit = chess.engine.Limit(time=0.1)
print('before: ', limit)
foo = engine.play(board, limit)
print('result foo:', foo)
print('result foo:', foo.move)
print('result foo/type:', type(foo.move))
print('result foo:', foo.ponder)
print('result foo:', foo.info)
# help(foo)
#sys.exit(0)

#
print('more')
moves = []
limit = chess.engine.Limit(depth=1)
board = chess.Board('r1bq1b1r/pppnpkpp/8/3n4/3P4/8/PPP2PPP/RNBQKB1R w KQ - 0 7')
board = chess.Board()
t1 = time.time()
while board.outcome() is None:
  ply = len(moves)
  rnd = ''
  if ply % 7 == 0:
    rnd = '*'
    move = random.choice(list(board.legal_moves))
  else:
    foo = engine.play(board, limit)
    move = foo.move
  uci = str(move)
  san = board.san(move)
  #print(uci, san, move, rnd)
  moves.append(san)
  board.push(move)
print('dt: ', time.time() - t1)
print('moves: ', moves)
print('final: ', board.outcome())
print(board)
print(board.fen)
print()
col = 0
strs = []
for i, san in enumerate(moves):
  if i % 2 == 0:
    s = f' {int(1+(i/2))}.'
    strs.append(s)
    col += len(s)
  strs.append(' ' + san)
  col += 1 + len(san)
  if col >= 72:
    print(''.join(strs))
    col = 0
    strs = []

engine.quit()
