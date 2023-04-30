import sys
import snappy
import struct
import tensorflow as tf

f = open('mega-v2-1.snappy', 'rb')

while True:
  n = struct.unpack('@i', f.read(4))[0]
  print(n)
  rest = f.read(n)
  big = snappy.uncompress(rest)
  #ex = snappy.uncompress(big)
  ex2 = tf.train.Example().FromString(big)
  print(ex2)
  break

