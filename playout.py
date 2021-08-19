import tensorflow as tf
import numpy as np
import pandas as pd
from absl import app
import sys, os
from absl import app
from absl import flags
from absl import logging
FLAGS = flags.FLAGS


# import open_spiel
import pyspiel

# from open_spiel.python.games import chess
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
  #help(game)

  print('game: ', game)
  state = game.new_initial_state()
  #help(state)
  legal = state.legal_actions_mask()
  print(len(legal))
  print(state.legal_actions())
  #sys.exit(0)  
  #help(state)
  print(state)
  print(state.is_terminal())
  #print(state.legal_actions())

  sans = []
  while not state.is_terminal():
    print()
    print('---------')    
    print('FEN: ', state) 
    board = make_observation(game).tensor.reshape(1, 20, 8, 8)
    #print('board: ', board)
    logits = model.predict([board])[0]
    print('logits: ', logits.shape)
    
    softmax = tf.nn.softmax(logits)
    print('softmax: ', softmax)
    besti = np.argmax(softmax, axis=0)


    #mask_softmax = np.multiply(softmax, np.array(state.legal_actions_mask()))
    mask_softmax = np.multiply(softmax, state.legal_actions_mask())
    print('besti: ', besti, ' ', softmax[besti], 'also',
          mask_softmax[besti])
          
    print('legal softmaxes: ', [softmax[i].numpy() for i in state.legal_actions()])
    print('legal2s:', [state.action_to_string(foo) for foo in state.legal_actions()])
    print('legal: ', state.legal_actions())
    print('legal nz: ', np.nonzero(state.legal_actions_mask()));    

    print('softmax_mask: ', mask_softmax)

    print('nz: ', 
          np.count_nonzero(logits),
          np.count_nonzero(softmax),
          np.count_nonzero(mask_softmax))

    choice = np.random.choice(range(4672), p=tf.nn.softmax(mask_softmax))
    print('choice: ', choice)
    print('xchoice: ', softmax[choice])
    print('xchoice: ', mask_softmax[choice])
    print('xchoice: ', state.legal_actions_mask()[choice])
    sys.exit(0)    
    
    if True:
      print('new soft', tf.nn.softmax(mask_softmax))
      print('samples: ', [np.random.choice(range(4672), p=tf.nn.softmax(mask_softmax)) for _ in range(10)])
      best = np.random.choice(range(4672), p=tf.nn.softmax(mask_softmax))
    else:
      best = np.argmax(mask_softmax, axis=0)
    print('best: ', best)
    print('legal: ', state.legal_actions())

    san = state.action_to_string(best)
    print('a2s: ', san)
    sans.append(san)

    state.apply_action_with_legality_check(best)
  print()
  print(' '.join(sans))


    #pprint('game: ', game)  
  
  


    
if __name__ == '__main__':
  app.run(main)      
