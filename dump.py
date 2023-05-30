import zlib
import pickle
import sqlitedict

def my_encode(obj):
  return zlib.compress(pickle.dumps(obj, pickle.HIGHEST_PROTOCOL))


def my_decode(obj):
  return pickle.loads(zlib.decompress(bytes(obj)))

dbio = sqlitedict.open('foo.sqlite',
                       flag='r',
                       timeout=60,
                       encode=my_encode,
                       decode=my_decode)

for k, v in dbio.items():
  print(k)
  print(type(v))
  print(len(v.SerializeToString()))
  print(len(my_encode(v)))
  print('level 6', len(zlib.compress(pickle.dumps(v, pickle.HIGHEST_PROTOCOL), level=6)))
  print('level 1', len(zlib.compress(pickle.dumps(v, pickle.HIGHEST_PROTOCOL), level=1)))
  print('level 9', len(zlib.compress(pickle.dumps(v, pickle.HIGHEST_PROTOCOL), level=9)))
  break
