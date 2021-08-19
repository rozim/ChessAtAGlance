import tensorflow as tf
import numpy as np
import pandas as pd
from absl import app
import sys, os
from absl import app
from absl import flags
from absl import logging
FLAGS = flags.FLAGS


import pyspiel

from open_spiel.python.observation import make_observation

FLAGS = flags.FLAGS

flags.DEFINE_string('plan', None, 'toml file')
flags.DEFINE_string('model', None, '')


def main(argv):
  flags.mark_flags_as_required(['model'])
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  
  model = tf.keras.models.load_model(FLAGS.model)
  
  game = pyspiel.load_game('chess')

  print('game: ', game)
  state = game.new_initial_state()

  legal = state.legal_actions_mask()
  print(state)
  print(state.is_terminal())

  sans = []
  while not state.is_terminal():
    print()
    print('---------')    
    print('FEN: ', state) 
    board = make_observation(game).tensor.reshape(1, 20, 8, 8)

    logits = model.predict([board])[0]
    
    logits2 = []
    i2a = []
    for i, a in enumerate(state.legal_actions()):
      logits2.append(logits[a])
      i2a.append(a)

    softmax2 = tf.nn.softmax(np.array(logits2, dtype='float32')).numpy()
    softmax2 /= softmax2.sum() # try to get closer to 1.0

    choice_action = np.random.choice(i2a, p=softmax2)
    san = state.action_to_string(choice_action)
    print('a2s: ', san)
    sans.append(san)

    state.apply_action_with_legality_check(choice_action)
  print()

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
