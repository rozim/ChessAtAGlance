
# Input: leveldb
# Output: TFRecord

import sys, os
import leveldb
import time
import struct
import snappy

def main(argv):  
  db = leveldb.LevelDB(
  n = 0
  tot = 0
  for ent in db.RangeIter():
    n += 1
    tot += len(ent[1])
    if n > lim:
      break
  db = None
  return n, tot



  plan = load_plan('v0.toml')
  print(next(iter(create_input_generator(plan.data, 'mega-v2-9.snappy'))))

    
if __name__ == '__main__':
  app.run(main)    
