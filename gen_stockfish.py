import chess
import chess.pgn
import chess.engine
import sys, os
import random
import time

from absl import app
from absl import flags
from absl import logging
from random import random, choice

FLAGS = flags.FLAGS
flags.DEFINE_integer('n', 10, '')


STOCKFISH = '/opt/homebrew/bin/stockfish'


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
    else:
      foo = engine.play(board, limit)
      move = foo.move

      if last_random or random() < pct_example:
        yield ply, board.fen(), move
      last_random = False        
      
    board.push(move)


def main(argv):
  ecos = list(parse_eco_fen())
  ecos.append('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')

  engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)

  nf = 0
  all_fens = set()
  while nf < FLAGS.n:
    for ply, fen, move in play1(engine,
                                choice(ecos),
                                chess.engine.Limit(depth=1),
                                pct_random=0.10,
                                pct_example=0.10):
      sfen = ' '.join(fen.split(' ')[0:3])
      if sfen in all_fens:
        continue
      all_fens.add(sfen)
      print(f'{fen},{move}')
      nf += 1
      if nf >= FLAGS.n:
        break
    
  engine.quit()

  
if __name__ == '__main__':
  app.run(main)      
