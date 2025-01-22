import glob
import io
import json
import os
import random
import sys
import time
import zlib

import chess
import chess.pgn
import jsonlines
import numpy as np
from absl import app, flags, logging
from chess import BLACK, WHITE

from encode import encode_cnn_board_move_wtm

FLAGS = flags.FLAGS
flags.DEFINE_string('pgn', 't1.pgn', 'PGN file or pattern')
flags.DEFINE_string('out', '', 'Output, jsonlines')

RESULT_MAP = {'1-0': 0,
              '1/2-1/2': 1,
              '0-1': 2}

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

def munch(pgn, fp, already):
  ng = 0
  nm = 0
  with jsonlines.Writer(fp, sort_keys=True) as writer:
    for i, game in enumerate(gen_games(pgn)):
      headers = game.headers
      try:
        iresult = RESULT_MAP[headers['Result']] # 0/1/2 WDL
      except KeyError:
        continue  # Result: "*" presumably
      ng += 1
      moves = list(gen_moves(game))
      for j, (move, san, sfen, ply) in enumerate(moves):
        nm += 1
        if sfen in already:
          continue
        board = chess.Board(sfen)
        enc_board, py_move = encode_cnn_board_move_wtm(board, move)

        assert enc_board.shape == (16, 64)  # CNN_SHAPE_2D
        py_board = enc_board.astype(int).flatten().tolist()
        assert len(py_board) == (1024), len(py_board) # len(INDEX_TO_MOVE)
        assert 0 <= py_move < 1968, py_move

        result = iresult if board.turn == WHITE else 2 - iresult
        j = {'sfen': sfen,
             'board_1024': py_board,
             'move_1968': py_move,
             'move_uci': move.uci(),
             'move_san': san,
             'result_3': result,
             'ply': j,
             'remain': len(moves) - j}
        writer.write(j)

  return ng, nm

def main(_):


  assert os.path.exists(FLAGS.pgn)
  assert 'jsonl' in FLAGS.out

  with open(FLAGS.out, 'w') as fp:
    t1 = time.time()
    ng, nm = munch(FLAGS.pgn, fp, set())
    dt = time.time() - t1
    print(f'{ng} {nm} | gps={ng/dt:.1f} mps={nm/dt:.1f} {dt:.1f}s')


if __name__ == '__main__':
  app.run(main)
