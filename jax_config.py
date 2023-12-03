from ml_collections import config_dict

from absl import app
from absl import flags
from absl import logging


def get_config() -> config_dict.ConfigDict:
  config = config_dict.ConfigDict()
  config.train = config_dict.ConfigDict()
  config.train.data = config_dict.ConfigDict()
  config.test = config_dict.ConfigDict()
  config.test.data = config_dict.ConfigDict()

  config.model = config_dict.ConfigDict()

  config.lr = 5e-3
  config.batch_size = 1024
  config.epochs = 1000

  config.train.steps = 10
  config.test.steps = 5

  config.train.data.batch_size = config.get_ref('batch_size')
  config.test.data.batch_size = config.get_ref('batch_size')

  config.train.data.shuffle = 100 * config.get_ref('batch_size')
  config.test.data.shuffle = 0

  config.train.data.pat = 'data/cnn-1m-0000[0-8]-of-00010.recordio'
  config.test.data.pat = 'data/cnn-1m-0000[9]-of-00010.recordio'

  config.model_type = 'cnn' # 'bias'
  config.model.num_blocks = 1
  config.model.num_filters = 64

  config.model.num_top = 1
  config.model.top_width = 1024

  return config


def main(argv):
  print(get_config())

if __name__ == "__main__":
  app.run(main)
