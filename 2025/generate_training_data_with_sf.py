# Use Stockfish to generate training data.
# Read in PGN file of presumably serious games.
# Make all moves in every position and search these positions.
# Adds "noise" or variety to eventual training data.

import glob
import io
import json
import os
import random
import sys
import time
import zlib

import chess
import chess.engine
import chess.pgn
import jsonlines
import numpy as np
from absl import app, flags, logging
from chess import BLACK, WHITE

from encode import encode_cnn_board_move_wtm

FLAGS = flags.FLAGS
flags.DEFINE_string('pgn', 't1.pgn', 'PGN file or pattern')
flags.DEFINE_string('out', '', 'Output, jsonlines')
flags.DEFINE_string('engine', './stockfish', '')
flags.DEFINE_integer('search_depth', 6, '')
flags.DEFINE_integer('multipv', 0, '')
flags.DEFINE_integer('delta', 0, 'For when using multipv')

HASH = 1024
THREADS = 1


def gen_games(fn):
  with open(fn, encoding="ISO-8859-1") as pgn:
    while True:
      g = chess.pgn.read_game(pgn)
      if g is None:
        return
      yield g


def gen_moves(game):
  board = game.board()
  for ply, move in enumerate(game.mainline_moves()):
    yield move, board.san(move), simplify_fen(board), ply
    board.push(move)


def simplify_fen(board):
  #rn2kbnr/ppq2pp1/4p3/2pp2Bp/2P4P/1Q6/P2NNPP1/3RK2R w Kkq - 2 13
  return ' '.join(board.fen().split(' ')[0:4])


def munch(pgn: str, fp, already: set, engine):
  # move1: game move
  # move2: all moves, from mml
  # move3: sf's best move

  ng = 0
  nm = 0
  with jsonlines.Writer(fp, sort_keys=True) as writer:
    for i, game in enumerate(gen_games(pgn)):
      ng += 1
      moves = list(gen_moves(game))
      for j, (move1, san1, sfen1, ply1) in enumerate(moves):
        nm += 1
        if sfen1 in already:
          continue
        already.add(sfen1)
        board = chess.Board(sfen1)
        for move2 in board.legal_moves:
          board.push(move2)
          if board.outcome() is not None:
            board.pop()
            continue
          sfen2 = simplify_fen(board)
          if sfen2 in already:
            board.pop()
            continue
          already.add(sfen2)

          if FLAGS.multipv == 1:
            res3 = engine.analyse(board, chess.engine.Limit(depth=FLAGS.search_depth))
            move3 = res3['pv'][0]
            san3 = board.san(move3)
            enc_board3, py_move3 = encode_cnn_board_move_wtm(board, move3)
            py_board3 = enc_board3.astype(int).flatten().tolist()

            # hack = chess.Board(sfen2)
            # assert hack.is_legal(move3), [hack.fen(), move3.uci()]
            # engine.quit()
            # sys.exit(0)
            j = {'sfen': sfen2,
                 'board_1024': py_board3,
                 'move_1968': py_move3,
                 'move_uci': move3.uci(),
                 'move_san': san3}
            writer.write(j)
          else:
            multi3 = engine.analyse(board, chess.engine.Limit(depth=FLAGS.search_depth), multipv=FLAGS.multipv)
            first = multi3[0]['score'].white()
            first_mate = first.is_mate()
            # if first.score() is None:
            #   print(board.fen())
            #   print(multi3)
            #   sys.exit(0)
            #   continue
            for i3, res3 in enumerate(multi3):
              white = res3['score'].white()
              try:
                if first_mate:
                  if not white.is_mate():
                    continue
                elif white.is_mate():
                  continue # confusing scenario?!
                elif abs(white.score() - first.score()) > FLAGS.delta:
                  continue
              except TypeError:
                import pprint
                print(res3['score'].white().is_mate())
                print(res3['score'].black().is_mate())
                print(res3['score'].white().mate())
                print(res3['score'].black().mate())
                print(first.mate())
                pprint.pprint(multi3)
                engine.quit()
                sys.exit(123)
              move3 = res3['pv'][0]

              san3 = board.san(move3)
              enc_board3, py_move3 = encode_cnn_board_move_wtm(board, move3)
              py_board3 = enc_board3.astype(int).flatten().tolist()
              j = {'sfen': sfen2,
                   'board_1024': py_board3,
                   'move_1968': py_move3,
                   'move_uci': move3.uci(),
                   'move_san': san3,
                   'multi': i3}
              writer.write(j)

          board.pop()

  return ng, nm


def main(_):
  assert os.path.exists(FLAGS.pgn)
  assert os.path.exists(FLAGS.engine)
  assert 'jsonl' in FLAGS.out

  engine = chess.engine.SimpleEngine.popen_uci(FLAGS.engine)
  engine.ping()
  engine.configure({'Hash': HASH})
  engine.configure({'Threads': THREADS})
  # res = engine.analyse(board, chess.engine.Limit(depth=1))

  with open(FLAGS.out, 'w') as fp:
    t1 = time.time()
    ng, nm = munch(FLAGS.pgn, fp, set(), engine)
    dt = time.time() - t1
    print(f'{ng} {nm} | gps={ng/dt:.1f} mps={nm/dt:.1f} {dt:.1f}s')
  engine.quit()


if __name__ == '__main__':
  app.run(main)
