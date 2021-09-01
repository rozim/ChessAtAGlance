import tensorflow as tf
import numpy as np
import pandas as pd

import sys, os
from absl import app
from absl import flags
from absl import logging
FLAGS = flags.FLAGS


import pyspiel

from open_spiel.python.observation import make_observation
from data import NUM_CLASSES
from data import create_input_generator_rio
from plan import load_plan

FLAGS = flags.FLAGS

flags.DEFINE_string('model', None, '')
flags.DEFINE_string('plan', None, '')
flags.DEFINE_string('fen', None, '')


def f2(f):
  return f'{f:.2f}'

def main(argv):
  flags.mark_flags_as_required(['model'])
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  plan = load_plan(FLAGS.plan)
  game = pyspiel.load_game('chess')
  ds = create_input_generator_rio(plan.data,
                                  ['mega-v5-9.recordio'],
                                  is_train=False, verbose=True, do_repeat=False, return_legal_moves=True)



  model = tf.keras.models.load_model(FLAGS.model, {'tf': tf,
                                                   'NUM_CLASSES': NUM_CLASSES})

  p2 = model.predict(ds, steps=1)
  print('p2: ', p2.shape)
  #sys.exit(0)
  #for ent in iter(ds):
    #print('ent=', ent)
    #print('p3', model.predict_on_batch(ent))
    #print('p1', model(ent))
    #break



  #game.deserialize_state('rnbqk2r/5pb1/p1pppnp1/1p5p/3P4/PPPBPNPP/5P2/RNBQK2R w KQkq - 1 10')

  #help(game)

  #print('game: ', game)
  #print(game.new_initial_state(fen).legal_actions())
  #print(game.new_initial_state().legal_actions()  )
  #fen = 'rnbqk2r/5pb1/p1pppnp1/1p5p/3P4/PPPBPNPP/5P2/RNBQK2R w KQkq - 1 10'
  if FLAGS.fen:
    state = game.new_initial_state(FLAGS.fen)
  else:
    state = game.new_initial_state()
  #help(state)

  print(state)

  #sys.exit(0)

  sans = []
  while not state.is_terminal():
    print()
    print('---------')
    print('FEN: ', state)
    print('REW: ', state.returns())
    board = make_observation(game).tensor.reshape(1, 20, 8, 8)
    legal_moves = np.array(state.legal_actions())
    legal_moves = np.reshape(legal_moves, (1, -1))
    legal_moves = tf.sparse.from_dense(legal_moves)
    logits = model.predict_on_batch(x=[board, [legal_moves]])
    logits = logits[0]
    softmax = tf.nn.softmax(np.array(logits, dtype='float32')).numpy()
    #softmax /= softmax.sum() # try to get closer to 1.0

    topn_idx = (-softmax).argsort()[:3]
    print('inx: ', topn_idx)
    scores = []
    for ent in topn_idx:
      scores.append(softmax[ent])
      print(ent, softmax[ent],
            state.action_to_string(ent))
    topn_scores = np.array(scores)
    topn_scores /= topn_scores.sum()
    choice_action = np.random.choice(topn_idx, p=topn_scores)
    san = state.action_to_string(choice_action)
    print('a2s: ', san)
    sans.append(san)

    state.apply_action_with_legality_check(choice_action)

  print()

  print('x0', state.is_terminal())
  print('x1', state.rewards())
  print('x2', state.returns())
  print('x3', game.utility_sum())

  col = 0
  buf = []
  for ply, san in enumerate(sans):
    if ply % 2 == 0:
      s = f' {int(1+(ply/2))}.'
      buf.append(s)
      col += len(s)
    buf.append(' ' + san)
    col += 1 + len(san)
    if col >= 72:
      print(''.join(buf))
      buf = []
      col = 0



# rnbqk2r/5pb1/p1pppnp1/1p5p/3P4/PPPBPNPP/5P2/RNBQK2R w KQkq - 1 10

if __name__ == '__main__':
  app.run(main)
