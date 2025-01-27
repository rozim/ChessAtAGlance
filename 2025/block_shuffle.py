# /opt/homebrew/bin/shuf seems insanely slow
# let's try an approximation.
# First draft from chatgpt.

import gzip
import os
import random

from absl import app, flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input', '', '*.txt or *.txt.gz')
flags.DEFINE_string('output', '', '*.txt or *.txt.gz')
flags.DEFINE_integer('buffer_size', 1_000_000, '')

def smart_output(fn, mode='w'):
  if fn.endswith('.gz'):
    return gzip.open(fn, mode)
  return open(fn, mode)


def smart_input(fn, mode='r'):
  if fn.endswith('.gz'):
    return gzip.open(fn, mode)
  return open(fn, mode)

def shuffle_large_file(input_file: str, output_file: str, buffer_size=1000):
  """
    Shuffles lines in a very large text file using a buffer.

    Parameters:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output shuffled file.
        buffer_size (int): Number of lines to hold in memory at a time.
    """
  buffer = []

  with smart_input(FLAGS.input, 'rt') as infile, smart_output(FLAGS.output, 'w') as outfile:
    # Fill the buffer with the initial `buffer_size` lines
    for _ in range(buffer_size):
      buffer.append(infile.readline())

    # Process remaining lines
    for line in infile:
      random_index = random.randint(0, len(buffer) - 1)
      outfile.write(buffer[random_index])
      buffer[random_index] = line

    # Write out remaining lines in the buffer
    random.shuffle(buffer)
    outfile.writelines(buffer)



def main(_):
  assert FLAGS.input
  assert FLAGS.output
  assert FLAGS.input != FLAGS.output
  assert os.path.exists(FLAGS.input)

  shuffle_large_file(FLAGS.input, FLAGS.output, FLAGS.buffer_size)

if __name__ == '__main__':
  app.run(main)
