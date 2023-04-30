import tensorflow as tf
import numpy as np
import pandas as pd

# import open_spiel
import pyspiel

# from open_spiel.python.games import chess
from open_spiel.python.observation import make_observation

game = pyspiel.load_game('chess')
print(game)
state = game.new_initial_state()
print(state)
print(state.is_terminal())
print(state.legal_actions())
