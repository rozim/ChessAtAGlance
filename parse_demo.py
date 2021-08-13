import sys
import tensorflow as tf
import leveldb
import snappy

# db = leveldb.LevelDB('mega-v2-1.leveldb')
# ent = next(db.RangeIter())
# print(type(ent[1]))
# print(len(ent[1]))
# print(len(snappy.compress(ent[1])))
# print(db.GetStats())
# sys.exit(0)

# db = leveldb.LevelDB('gen.leveldb')

# ent = next(db.RangeIter())

# INPUT_SHAPE = (20, 8, 8)
# feature_spec = { 'board': tf.io.FixedLenFeature(INPUT_SHAPE, tf.dtypes.float32)}

# print(type(ent[1]))
# ex = tf.train.Example().FromString(ent[1])
# print(len(ent[1]))
# print(type(ent[1]))
# print(type(ex.features.feature['board']))
# # help(ex.features.feature['board'])
# #b = tf.train.batch(ent[1], batch_size=1)
# #p = tf.io.parse_example(serialized=ent[1], features=feature_spec)

# #p = tf.io.parse_single_example(serialized=ex, features=feature_spec)
# foo = ex.features.feature['board'].float_list.value
# print(tf.convert_to_tensor(foo))

def gen():
  db = leveldb.LevelDB('mega-v2-1.leveldb') 
  for ent in db.RangeIter():
    ex = tf.train.Example().FromString(ent[1])
    #print(ex)
    board = tf.convert_to_tensor(ex.features.feature['board'].float_list.value,
                                 dtype=tf.float32)
    action = tf.convert_to_tensor(ex.features.feature['label'].int64_list.value[0],
                                 dtype=tf.int64)
    #print(action)
    #sys.exit(0)
    yield (board, action)


ds = tf.data.Dataset.from_generator(gen,
                                    output_types=('float32', 'int64'),
                                    output_shapes=([1280,], []))

foo = next(iter(ds))
print(foo)
# print(foo)
# print(foo[0])
# print(foo[1])
# print(foo[1].numpy())



