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
import os, sys
import random
import time
import glob

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
flags.DEFINE_integer('shards', 1, 'Number of shards')
# flags.mark_flag_as_required('pgn')
# flags.mark_flag_as_required('out')

def shuffled(ar):
  random.shuffle(ar)
  return ar


def main(argv):
  assert FLAGS.pgn
  assert FLAGS.out
  assert FLAGS.shards > 0

  opts = tf.io.TFRecordOptions(
    compression_type='ZLIB',
    output_buffer_size=(4 * 1024 * 1024))

  if FLAGS.shards == 1:
    rio = [tf.io.TFRecordWriter(FLAGS.out, opts)]
  else:
    rio = [
      tf.io.TFRecordWriter(f'{FLAGS.out}-{shard:05d}-of-{FLAGS.shards:05d}', opts)
      for shard in range(FLAGS.shards)]

  already = set()
  n_game, n_move, n_gen, n_dup = 0, 0, 0, 0
  mod = 1
  files = shuffled(glob.glob(FLAGS.pgn))
  for fnum, pgn_fn in enumerate(files):
    print(f'Open: {fnum}/{len(files)}: {pgn_fn}')
    for i, game in enumerate(gen_games(pgn_fn)):
      n_game += 1
      if n_game % mod == 0:
        print(n_game, n_move, n_gen, n_dup)
        mod *= 2
        mod = min(mod, 1024)

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
        if n_gen > 25000000:
          # Else we use up all RAM.
          print('RESET')
          already = set()

        enc_board, enc_move = encode_cnn_board_move_wtm(board, move)
        feature = {
          'board': floats_feature(enc_board.flatten()),
          'label': int64_feature(enc_move),
          'fen':  bytes_feature(fen.encode('utf-8')),
        }
        pb = tf.train.Example(features=tf.train.Features(feature=feature))
        rio[random.randint(0, FLAGS.shards-1)].write(pb.SerializeToString())

  for fh in rio:
    fh.close()

  print('n_game: ', n_game)
  print('n_move: ', n_move)
  print('n_dup: ', n_dup)
  print('n_gen: ', n_gen)



if __name__ == '__main__':
  app.run(main)
