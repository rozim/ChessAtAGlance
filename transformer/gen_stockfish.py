# Generate training data with stockfish.
# Start off in common ECO positions.
#
# Usually make the best move but sometimes make
# a random move (PCT_RANDOM).
#
# Limit the max ply so that random moves by the stronger
# side don't prolong games forever.
#
# Only generate unique FENs.
# Approx 300 FEN/sec for depth=1.
#
# Sample moves, don't use all moves else there may be bias.
#
#
#

import secrets
import sys, os
import random
import time

import chess
import chess.engine
from chess import WHITE, BLACK


from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_integer('goal', 10, 'Positions to generate')
flags.DEFINE_integer('depth', 1, 'search depth for analysis we record')
flags.DEFINE_integer('game_depth', 1, 'game search depth to choose a move')
flags.DEFINE_integer('max_game_ply', 100, '')

STOCKFISH = './stockfish'
HASH = 512
THREADS = 1
PCT_RANDOM = 0.05
PCT_SAMPLE_RANDOM = 0.25
PCT_SAMPLE_BEST = 0.25
MAX_SAMPLES_PER_GAME = 10

max_game = 0
total_ply = 0


def secrets_random():
  # Just for kicks use crypto-secure random.
  return secrets.randbelow(1000) / 1000.0

def secrets_choice(lis):
  return secrets.choice(lis)


def parse_eco_fen():
  for what in ['a', 'b', 'c', 'd', 'e']:
    with open(f'eco/{what}.tsv', 'r') as f:
      first = True
      for line in f:
        if first:
          first = False
          continue
        yield line.split('\t')[2]


def read_eco():
  ecos = list(parse_eco_fen())
  ecos.append('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
  return ecos


def play2(engine, starting_fen, pct_random, pct_sample_random, pct_sample_best):
  global total_ply
  board = chess.Board(starting_fen)
  ply = -1
  samples = 0

  while board.outcome() is None:
    total_ply += 1
    ply += 1
    if ply >= FLAGS.max_game_ply:
      global max_game
      max_game += 1
      return

    if secrets_random() < pct_random:
      # To add variety, sometimes just move randomly and
      # don't analyze.
      move = secrets_choice(list(board.legal_moves))
      played_random = True
    else: # Play best
      engine.configure({"Clear Hash": None})
      res = engine.analyse(board, chess.engine.Limit(depth=FLAGS.game_depth))
      move = res['pv'][0]
      played_random = False
    board.push(move)

    if played_random:
      do_analyze = secrets_random() < pct_sample_random
    else:
      do_analyze = secrets_random() < pct_sample_best
    if do_analyze and board.outcome() is None:
      # Occasionally analyze the current position.
      do_mirror = (board.turn == BLACK)
      if do_mirror:
        mirror_mul = -1 # Make score WTM POV.
        board2 = board.copy().mirror()
      else:
        mirror_mul = 1
        board2 = board.copy()
      engine.configure({"Clear Hash": None})
      res = engine.analyse(board2, chess.engine.Limit(depth=FLAGS.depth))
      uci_move = res['pv'][0]
      san_move = board2.san(uci_move)
      yield simplify_fen(board2), uci_move, san_move, mirror_mul * simplify_score2(res['score'])[-1]
      samples += 1
      if samples >= MAX_SAMPLES_PER_GAME:
        return




def simplify_fen(board):
  #rn2kbnr/ppq2pp1/4p3/2pp2Bp/2P4P/1Q6/P2NNPP1/3RK2R w Kkq - 2 13
  return ' '.join(board.fen().split(' ')[0:4])


def simplify_score2(score):
  mx = 10000
  lim = 9000
  res = int(score.pov(chess.WHITE).score(mate_score=10000))
  if score.is_mate(): # normal, mate
    assert res > lim or res < -lim, 'mate in 1000 considered unlikely'
    return True, res
  elif res > lim: # clamp
    return False, lim
  elif res < -lim: # clamp
    return False, -lim
  else: # normal, in range
    return False, res


def main(argv):
  t1 = time.time()
  ecos = read_eco()

  engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
  engine.configure({"Hash": HASH})
  engine.configure({"Threads": THREADS})

  all_fens = set()
  dups = 0
  games = 0
  go_on = True

  while go_on:
    games += 1
    for fen, uci_move, san_move, score in play2(engine, secrets_choice(ecos),
                                                pct_random=PCT_RANDOM,
                                                pct_sample_random=PCT_SAMPLE_RANDOM,
                                                pct_sample_best=PCT_SAMPLE_BEST):
      if fen in all_fens:
        dups += 1
        continue
      all_fens.add(fen)
      print(f'{fen},{uci_move},{san_move},{score}')
      if len(all_fens) >= FLAGS.goal:
        go_on = False
        break

  engine.quit()
  sys.stderr.write(f'Too long: {max_game}\n')
  sys.stderr.write(f'Dups:     {dups}\n')
  sys.stderr.write(f'Games:    {games}\n')
  sys.stderr.write(f'Ply:      {total_ply}\n')
  sys.stderr.write(f'FENS:     {len(all_fens)}\n')

if __name__ == '__main__':
  app.run(main)


# formerly: exhaustive moving
#
# # Make every move, even complete garbage, and analyze, so that
# # the ML model learns to make obvious moves/recaptures etc.
# for move in board.legal_moves:
#   board.push(move)
#   if board.outcome() is None:
#     do_mirror = (board.turn == BLACK)
#     if do_mirror:
#       mirror_mul = -1 # Make score WTM POV.
#       board2 = board.copy().mirror()
#     else:
#       mirror_mul = 1
#       board2 = board.copy()
#     res = engine.analyse(board2, chess.engine.Limit(depth=FLAGS.depth))
#     uci_move = res['pv'][0]
#     san_move = board2.san(uci_move)
#     yield simplify_fen(board2), uci_move, san_move, mirror_mul * simplify_score2(res['score'])[-1]
#   board.pop()
