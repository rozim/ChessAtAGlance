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
  print(' '.join(sans))


if __name__ == '__main__':
  app.run(main)      
