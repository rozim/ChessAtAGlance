#
# Generate CNN-like training data with a board representation inspired by AlphaGo/Zero chess.
#
#
# Input: PGN files
#
# Output: RecordIO files of tf.Example
#
# Schema:
#   board:
#   move:
#
# Algorithm:
#
#
#
import os, sys, io
import random
import time

from absl import app
from absl import flags
from absl import logging

import chess
import chess.pgn

import numpy as np

import tensorflow as tf

from encode import encode_cnn_board_move_wtm
from pychess_util import *
from tf_util import *

FLAGS = flags.FLAGS

flags.DEFINE_string('pgn', 't1.pgn', 'PGN file')
flags.DEFINE_string('out', 'foo.rio', 'Recordio file')
# flags.mark_flag_as_required('pgn')
# flags.mark_flag_as_required('out')

def main(argv):
  assert FLAGS.pgn
  assert FLAGS.out

  opts = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    output_buffer_size=(4 * 1024 * 1024))

  already = set()
  n_game, n_move, n_gen, n_dup = 0, 0, 0, 0
  mod = 1
  with tf.io.TFRecordWriter(FLAGS.out, opts) as rio:
    for i, game in enumerate(gen_games(FLAGS.pgn)):
      n_game += 1
      if n_game % mod == 0:
        print(n_game, n_move, n_gen, n_dup)
        mod *= 2

      for ply, (move, board) in enumerate(gen_moves(game)):
        n_move += 1
        #print(i, ply, move)
        fen = simplify_fen(board)
        uci = move.uci()
        key = hash(fen + uci)
        if key in already:
          n_dup += 1
          continue
        already.add(key)
        n_gen += 1

        enc_board, enc_move = encode_cnn_board_move_wtm(board, move)
        feature = {
          'board': floats_feature(enc_board.flatten()),
          'label': int64_feature(enc_move),
          'fen':  bytes_feature(fen.encode('utf-8')),
        }
        pb = tf.train.Example(features=tf.train.Features(feature=feature))
        rio.write(pb.SerializeToString())
  print('n_game: ', n_game)
  print('n_move: ', n_move)
  print('n_dup: ', n_dup)
  print('n_gen: ', n_gen)



if __name__ == '__main__':
  app.run(main)
