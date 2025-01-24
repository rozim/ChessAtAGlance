# Use Stockfish to generate training data.
# Read in PGN file of presumably serious games.
# Make all moves in every position and search these positions.
# Adds "noise" or variety to eventual training data.

import os
import sys
import time

import chess
import chess.engine
import chess.pgn
import jsonlines
from absl import app, flags

from encode import encode_cnn_board_move_wtm
from contextlib import contextmanager
from typing import Any

FLAGS = flags.FLAGS
flags.DEFINE_string('input_pgn', '', 'PGN file or pattern')
flags.DEFINE_string('input_jsonl', '', 'Jsonl from previous run')
flags.DEFINE_string('output', '', 'Output, jsonlines')
flags.DEFINE_string('engine', './stockfish', '')
flags.DEFINE_integer('search_depth', 6, '')
flags.DEFINE_integer('multipv', 1, '')
flags.DEFINE_integer('delta', 0, 'For when using multipv')

HASH = 1024
THREADS = 1


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
    enc_board3, py_move3 = encode_cnn_board_move_wtm(board, move3)
    py_board3 = enc_board3.astype(int).flatten().tolist()
    yield {'sfen': sfen2,
           'board_1024': py_board3,
           'move_1968': py_move3,
           'move_uci': move3.uci(),
           'move_san': san3,
           'multi': i3}



def gen_games_moves(pgn: str):
  for game in gen_games(pgn):
    yield from gen_moves(game)


def gen_positions_for_sf(pgn: str, already: set):
  for (_, san1, sfen1, _) in gen_games_moves(pgn):
    if sfen1 in already:
      continue
    already.add(sfen1)
    yield sfen1  # Might as well analyze game position too so SF can be more thorough.

    # Now expand all moves and yield resulting positions.
    board = chess.Board(sfen1)
    for move2 in board.legal_moves:
      with push_pop_move(board, move2) as _:
        if board.outcome() is not None:
          continue
        sfen2 = simplify_fen(board)
        if sfen2 in already:
          continue
        already.add(sfen2)
        yield sfen2

def gen_positions_for_sf_from_fen(sfen1: str, already: set):
  already.add(sfen1)

  # Now expand all moves and yield resulting positions.
  board = chess.Board(sfen1)
  for move2 in board.legal_moves:
    with push_pop_move(board, move2) as _:
      if board.outcome() is not None:
        continue
      sfen2 = simplify_fen(board)
      if sfen2 in already:
        continue
      already.add(sfen2)
      yield sfen2



def main(_):
  if FLAGS.input_pgn:
    assert os.path.exists(FLAGS.input_pgn)
    assert 'pgn' in FLAGS.input_pgn
  if FLAGS.input_jsonl:
    assert os.path.exists(FLAGS.input_jsonl)
    assert 'jsonl' in FLAGS.input_jsonl
  assert FLAGS.input_pgn or FLAGS.input_jsonl
  assert not (FLAGS.input_pgn and FLAGS.input_jsonl) # whew

  assert 'jsonl' in FLAGS.output

  assert FLAGS.input_pgn != FLAGS.output
  assert FLAGS.input_jsonl != FLAGS.output

  assert os.path.exists(FLAGS.engine)

  engine = chess.engine.SimpleEngine.popen_uci(FLAGS.engine)
  engine.ping()
  engine.configure({'Hash': HASH})
  engine.configure({'Threads': THREADS})

  already = set()
  with open(FLAGS.output, 'w') as fp:
    with jsonlines.Writer(fp, sort_keys=True) as writer:
      if FLAGS.input_pgn:
        for fen in gen_positions_for_sf(FLAGS.input_pgn, already):
          for j in sf_analyze(engine, fen, depth=FLAGS.search_depth,
                              multipv=FLAGS.multipv,
                              delta=FLAGS.delta):
            writer.write(j)
      else:
        t1 = time.time()
        mod = 1000
        rows = 0
        print('Initial read')
        for j_in in jsonlines.open(FLAGS.input_jsonl, mode='r'):
          rows += 1
          already.add(j_in['sfen'])
          if rows % mod == 0:
            print(f'{len(already)} {rows} {time.time() - t1:.1f}')
            mod *= 2
            if mod > 10_0000:
              mod = 10_0000

        print(f'Already: {len(already)} {time.time() - t1}s')
        #for j_in in jsonlines.open(FLAGS.input_jsonl, mode='r'):
        row1, row2, row3 = 0, 0, 0
        mod = 1000
        t1 = time.time()
        for sfen1 in already.copy():
          row1 += 1
          if row1 % mod == 0:
            print(row1, row2, row3, time.time() - t1)
            mod *= 2
            if mod > 10_000:
              mod = 10_000
          for fen in gen_positions_for_sf_from_fen(sfen1, already):
            row2 += 1
            for j in sf_analyze(engine, fen, depth=FLAGS.search_depth,
                                multipv=FLAGS.multipv,
                                delta=FLAGS.delta):
              row3 += 1
              writer.write(j)

  engine.quit()


if __name__ == '__main__':
  app.run(main)
