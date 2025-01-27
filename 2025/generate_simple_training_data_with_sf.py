# Use Stockfish to generate training data.
# Read in FEN file of all moves.

import os
import sys
import time
##### from contextlib import contextmanager
from typing import Any
import gzip

import chess
import chess.engine
##### import chess.pgn
import jsonlines
from absl import app, flags

##### from encode import encode_cnn_board_move_wtm

FLAGS = flags.FLAGS
flags.DEFINE_string('input', '', 'FEN')
flags.DEFINE_string('output', '', 'Output, jsonlines, optionally .gz')
flags.DEFINE_string('engine', './stockfish', '')
flags.DEFINE_integer('search_depth', 1, '')
flags.DEFINE_integer('multipv', 5, '')
flags.DEFINE_integer('delta', 25, 'For when using multipv')

HASH = 1024
THREADS = 1


def strip(gen):
  return (line.strip() for line in gen)

def simplify_fen(board):
  #rn2kbnr/ppq2pp1/4p3/2pp2Bp/2P4P/1Q6/P2NNPP1/3RK2R w Kkq - 2 13
  return ' '.join(board.fen().split(' ')[0:4])


def sf_analyze(engine: Any, sfen2: str, *, depth: int, multipv: int, delta: int):
  board = chess.Board(sfen2)
  multi3 = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
  first = multi3[0]['score'].white()
  for i3, res3 in enumerate(multi3):
    white = res3['score'].white()
    if white.is_mate() or first.is_mate():
      continue
    elif abs(white.score() - first.score()) > delta:
      continue

    move3 = res3['pv'][0]
    san3 = board.san(move3)
    # enc_board3, py_move3 = encode_cnn_board_move_wtm(board, move3)
    # py_board3 = enc_board3.astype(int).flatten().tolist()
    yield {'sfen': sfen2,
           # 'board_1024': py_board3,
           # 'move_1968': py_move3,
           # 'move_uci': move3.uci(),
           'move_san': san3,
           # 'multi': i3
           }



def smart_output(fn, mode: str='wt'):
  if fn.endswith('.gz'):
    return gzip.open(fn, mode)
  return open(fn, mode)


def smart_input(fn, mode: str='rt'):
  if fn.endswith('.gz'):
    return gzip.open(fn, mode)
  return open(fn, mode)


def main(_):
  assert FLAGS.input
  assert os.path.exists(FLAGS.input)
  assert 'fen' in FLAGS.input

  assert FLAGS.output
  assert not os.path.exists(FLAGS.output)
  assert 'jsonl' in FLAGS.output

  assert FLAGS.input != FLAGS.output

  assert os.path.exists(FLAGS.engine)

  engine = chess.engine.SimpleEngine.popen_uci(FLAGS.engine)
  engine.ping()
  engine.configure({'Hash': HASH})
  engine.configure({'Threads': THREADS})

  n_in = 0
  n_out = 0
  with smart_output(FLAGS.output) as fp:
    with jsonlines.Writer(fp, sort_keys=True) as writer:
      for sfen in strip(smart_input(FLAGS.input, 'rt')):
        n_in += 1
        for j in sf_analyze(engine, sfen,
                            depth=FLAGS.search_depth,
                            multipv=FLAGS.multipv,
                            delta=FLAGS.delta):
          writer.write(j)
          n_out += 1

  engine.quit()
  print('In:  ', n_in)
  print('Out: ', n_out)
  sys.exit(0)


if __name__ == '__main__':
  app.run(main)
