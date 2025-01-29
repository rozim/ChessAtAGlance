
import sys
import os
import hashlib

from absl import app, flags

def main(argv):
  already = set()
  n = 0
  mod = 1
  for line in sys.stdin.readlines():
    sha256 = hashlib.sha256()
    sha256.update(line.encode('utf-8'))
    # already.add(int(sha256.hexdigest(), 16))
    already.add(sha256.digest())
    n += 1
    if n % mod == 0:
      print(n, len(already))
      mod *= 2

  print()
  print(n, len(already))

if __name__ == '__main__':
  app.run(main)
