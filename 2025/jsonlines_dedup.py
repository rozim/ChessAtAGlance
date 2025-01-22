

# Dedup board,move

import time

import jsonlines
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input',
                    'data/mega2600_shuffled.jsonl',
                    'Input - uncompressed json lines')
flags.DEFINE_string('output',
                    'data/mega2600_shuffled_dedup.jsonl',
                    '')



def main(argv):
  time.time()
  already = set()

  keep, reject = 0, 0
  fo = open(FLAGS.output, 'w')
  with jsonlines.Writer(fo, sort_keys=True) as writer:
    for j in jsonlines.open(FLAGS.input, 'r'):
      sfen = j['sfen']
      move_san = j['move_san']
      key = f'{move_san}|{sfen}'
      if key in already:
        reject += 1
      else:
        keep += 1
        writer.write(j)
        already.add(key)
  print('reject: ', reject)
  print('keep  : ', keep)
  fo.close()





if __name__ == '__main__':
  app.run(main)
