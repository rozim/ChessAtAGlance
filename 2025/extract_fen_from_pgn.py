# Read in PGN file of presumably serious games.
# Extract all unique FENs.

import os
import sys
import time
from contextlib import contextmanager
from typing import Any
import gzip

import chess
import chess.engine
import chess.pgn
import jsonlines
from absl import app, flags

from encode import encode_cnn_board_move_wtm

FLAGS = flags.FLAGS
flags.DEFINE_string('input', '', 'PGN file or pattern')
flags.DEFINE_string('output', '', 'Output, jsonlines, optionally .gz')


@contextmanager
def push_pop_move(board: chess.Board, move: chess.Move):
  try:
    board.push(move)
    yield
  finally:
    board.pop()


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


def gen_games_moves(pgn: str):
  for game in gen_games(pgn):
    yield from gen_moves(game)  # move, san, sfen, ply


def simplify_fen(board):
  #rn2kbnr/ppq2pp1/4p3/2pp2Bp/2P4P/1Q6/P2NNPP1/3RK2R w Kkq - 2 13
  return ' '.join(board.fen().split(' ')[0:4])


def smart_output(fn, mode='w'):
  if fn.endswith('.gz'):
    return gzip.open(fn, mode)
  return open(fn, mode)


def smart_input(fn, mode='r'):
  if fn.endswith('.gz'):
    return gzip.GzipFile(fn, mode)
  return open(fn, mode)


def main(_):
  assert os.path.exists(FLAGS.input)
  assert 'pgn' in FLAGS.input
  assert FLAGS.input != FLAGS.output
  assert FLAGS.output
  assert not os.path.exists(FLAGS.output), FLAGS.output

  already = set()

  with smart_output(FLAGS.output, 'wt') as fp:
    for _, _, sfen, _ in gen_games_moves(FLAGS.input):
      if sfen not in already:
        fp.write(sfen + '\n')


if __name__ == '__main__':
  app.run(main)
