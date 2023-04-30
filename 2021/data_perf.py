
import sys, os
import leveldb
import time
import struct
import snappy

def gen(fn, lim):
  db = leveldb.LevelDB(fn)
  n = 0
  tot = 0
  for ent in db.RangeIter():
    n += 1
    tot += len(ent[1])
    if n > lim:
      break
  db = None
  return n, tot


def gen2(fn, lim):
  f = open(fn, 'rb', 1024 * 1024)

  row = 0
  tot = 0
  uncompress = snappy.uncompress
  unpack = struct.unpack
  read = f.read
  while True:
    row += 1
    n = unpack('@i', f.read(4))[0]
    rest = read(n)
    big = uncompress(rest)
    tot += len(big)
    if row > lim:
      break
  f = None
  return row, tot

t1 = time.time()

#res = gen('mega-v2-0.leveldb',  2000000)
res = gen2('mega-v2-0.snappy', 1000000)



dt = time.time() - t1
print(f'dt {dt:.2f} {res}')
