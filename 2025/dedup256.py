# Try to dedup w/ sha256 instead of raw lines, possibly with "sort | uniq".

import sys
import os
import hashlib

from absl import app, flags

def main(argv):
  already = set()
  for line in sys.stdin.readlines():
    sha256 = hashlib.sha256()
    sha256.update(line.encode('utf-8'))
    key = sha256.digest()
    if key not in already:
      already.add(key)
      print(line, end='')


if __name__ == '__main__':
  app.run(main)
