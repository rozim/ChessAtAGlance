# Dedup that spills to disk.

import sys
import os
import tempfile
import gzip
import heapq
from absl import app, flags

CHUNK_SIZE = 1_000_000

def read_in_chunks(fp, chunk_size):
  """Generator to read a file in chunks."""
  ar = []
  for line in fp:
    ar.append(line.strip())
    if len(ar) == chunk_size:
      yield ar
      ar = []
  if len(ar):
    yield ar


def sort_and_write_chunk(chunk, temp_dir):
  """Sort a chunk and write it to a temporary file."""
  chunk = sorted(set(chunk))  # Sort and deduplicate
  temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, mode='w', suffix='-infinite.gz')
  temp_file.close()
  g = gzip.open(temp_file.name, 'wt')
  g.writelines(f"{line}\n" for line in chunk)
  g.close()
  return temp_file.name


def merge_sorted_files(file_list, output_file):
  """Merge multiple sorted files into one output file, removing duplicates."""
  with open(output_file, 'w', encoding='utf-8') as out:
    files = [gzip.open(file, 'rt') for file in file_list]
    iterators = [iter(file) for file in files]
    merged = heapq.merge(*iterators, key=str)

    prev = None
    for line in merged:
      line = line.strip()
      if line != prev:  # Deduplicate during merge
        out.write(line + '\n')
        prev = line

    for file in files:
      file.close()

  # Cleanup temporary files
  for file in file_list:
      os.remove(file)


def main(argv):
  temp_dir = tempfile.mkdtemp()
  sys.stderr.write(temp_dir + '\n')

  files = []
  for i, chunk in enumerate(read_in_chunks(sys.stdin, CHUNK_SIZE)):
    sys.stderr.write(f'Chunk {i} {len(chunk)} [{chunk[0]}]\n')
    fn = sort_and_write_chunk(chunk, temp_dir)
    files.append(fn)
  sys.stderr.write(f'Merging\n')
  merge_sorted_files(files, '/dev/stdout')
  os.rmdir(temp_dir)


if __name__ == '__main__':
  app.run(main)
