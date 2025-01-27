# Read in FEN file of presumably serious games.
# Expand all legal positions in WTM format.

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
flags.DEFINE_boolean('include', True, 'Include input FENs in output.')

@contextmanager
def push_pop_move(board: chess.Board, move: chess.Move):
  try:
    board.push(move)
    yield
  finally:
    board.pop()


def gen_legal_sfens(board: chess.Board):
  for board in gen_legal_boards(board):
    yield simplify_fen(board)


def gen_legal_boards(board: chess.Board):
  for move in board.legal_moves:
    board.push(move)
    if board.outcome() is None:
      yield board
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
    return gzip.open(fn, mode)
  return open(fn, mode)

def strip(gen):
  return (line.strip() for line in gen)

def main(_):
  assert os.path.exists(FLAGS.input)
  assert 'fen' in FLAGS.input
  assert 'pgn' not in FLAGS.input
  assert 'fen' in FLAGS.output
  assert 'pgn' not in FLAGS.output
  assert FLAGS.input != FLAGS.output
  assert FLAGS.output
  assert not os.path.exists(FLAGS.output), FLAGS.output

  already = set()

  with smart_output(FLAGS.output, 'wt') as fp:
    for sfen in strip(smart_input(FLAGS.input, 'rt')):
      assert sfen not in already, sfen
      assert chess.Board(sfen).turn == chess.WHITE
      already.add(sfen)
      if FLAGS.include:
        fp.write(sfen + '\n')

    for sfen in strip(smart_input(FLAGS.input, 'rt')):
      for sfen2 in gen_legal_sfens(chess.Board(sfen)):
        board2 = chess.Board(sfen2)
        if board2.turn == chess.BLACK:
          sfen2 = simplify_fen(board2.mirror())
        else:
          assert False, 'insanity' # assumed wtm
        if sfen2 in already:
          continue
        fp.write(sfen2 + '\n')
        already.add(sfen2)


if __name__ == '__main__':
  app.run(main)
