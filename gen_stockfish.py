import chess
import chess.engine
import sys, os
import random
import time

from absl import app
from absl import flags
from absl import logging
from random import random, choice

FLAGS = flags.FLAGS
flags.DEFINE_integer('n', 10, 'number')
flags.DEFINE_integer('d', 1, 'search depth')

STOCKFISH = '/opt/homebrew/bin/stockfish'

# Score threshold, pawn=100
THRESHOLD = 25

# Assume after this many moves that there is something strange
# about the position having too many near-optimal moves.
MAX_THRESHOLD_GEN = 5

def parse_eco_fen():
  for what in ['a', 'b', 'c', 'd', 'e']:
    with open(f'eco/{what}.tsv', 'r') as f:
      first = True
      for line in f:
        if first:
          first = False
          continue
        yield line.split('\t')[2]


def play1(engine, fen, limit, pct_random, pct_example):
  board = chess.Board(fen)

  last_random = False # avoid 2 in a row
  ply = -1
  while board.outcome() is None:
    ply += 1
    if not last_random and random() < pct_random:
      move = choice(list(board.legal_moves))
      last_random = True
      board.push(move)
      continue

    legal_moves = set(board.legal_moves)
    res1 = engine.play(board, limit, info=chess.engine.INFO_SCORE)

    score1 = res1.info['score'].relative.score()
    move1 = res1.move
    assert move1 in legal_moves

    if not last_random and random() >= pct_example:
      board.push(move1)
      continue

    yield ply, board.fen(), move1, score1
    last_random = False

    legal_moves.remove(move1)
    yes = 0
    while len(legal_moves) > 0 and yes < MAX_THRESHOLD_GEN:
      res2 = engine.play(board, limit, info=chess.engine.INFO_SCORE, root_moves=legal_moves)
      score2 = res2.info['score'].relative.score()
      move2 = res2.move
      assert move1 != move2
      assert move2 in legal_moves
      try:
        m1 = res1.info['score'].is_mate()
        m2 = res2.info['score'].is_mate()
        if m1 and m2:
          delta = 0
        elif m1 or m2:
          break
        else:
          delta = abs(score1 - score2)
      except TypeError:
        print('bug', move1, score1, move2, score2, legal_moves)
        print(board.fen())
        print(res1.info['score'].is_mate())
        help(score1)
        engine.quit()
        sys.exit(0)
      if delta > THRESHOLD:
        break
      yes += 1

      yield ply, board.fen(), move2, score2
      legal_moves.remove(move2)
    #print('*', yes)
    board.push(move1)



# play2 to generate scores
def play2(engine, fen, depths, pct_random):
  board = chess.Board(fen)

  ply = -1
  while board.outcome() is None:
    ply += 1
    if random() < pct_random:
      # To add variety, sometimes just move randomly and
      # don't analyze.
      move = choice(list(board.legal_moves))
      board.push(move)
    else:
      scores = []
      for depth in depths:
        res = engine.play(board, depth, info=chess.engine.INFO_SCORE)

        score = res.info['score']
        if score.is_mate():
          iscore = 9999
        else:
          iscore = score.relative.score()
        scores.append(iscore)

      yield ply, board.fen(), scores
      board.push(res.move) # use last one


def main(argv):
  ecos = list(parse_eco_fen())
  ecos.append('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

  engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

  nf = 0
  all_fens = set()
  dups = 0
  depths = [chess.engine.Limit(depth=d) for d in range(10)]
  while nf < FLAGS.n:
    for ply, fen, scores in play2(engine, choice(ecos), depths, pct_random=0.25):
      ar = fen.split(' ')
      sfen = ar[0] + ' ' + ar[1] + ' ' + ar[2]
      #print('sfen: ', sfen)
      if sfen in all_fens:
        #print('DUP')
        dups += 1
        continue
      all_fens.add(sfen)
      #print(f'{fen},{move}')
      pp = ','.join([f'{score}' for score in scores])
      print(f'{fen},{pp}')
      nf += 1
      if nf >= FLAGS.n:
        break

  engine.quit()

  print('fens: ', len(all_fens), dups)
  #print('table: ', all_fens)


if __name__ == '__main__':
  app.run(main)
