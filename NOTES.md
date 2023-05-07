2023-05-06
==========

how to import mish

from Mish.Mish.TFKeras import mish

how to import smelu



2023-05-03
==========

keras.initializers.glorot_normal
keras.initializers.glorot_uniform
keras.initializers.he_normal
keras.initializers.he_uniform
keras.initializers.lecun_normal
keras.initializers.lecun_uniform
keras.initializers.orthogonal
keras.initializers.random_normal
keras.initializers.random_uniform
keras.initializers.truncated_normal
keras.initializers.variance_scaling

3 runs of each initializer with filters=16, layers=4, top_tower=256, epochs=100,
from worst to best:

0.8479 : he_uniform-out.txt
0.8489 : he_normal-out.txt
0.8503 : glorot_uniform-out.txt
0.8528 : variance_scaling-out.txt
0.8536 : lecun_normal-out.txt
0.8558 : lecun_uniform-out.txt
0.8566 : truncated_normal-out.txt
0.8567 : glorot_normal-out.txt
0.8611 : random_uniform-out.txt
0.8615 : orthogonal-out.txt
0.8624 : random_normal-out.txt		<-- best

2023-05-02
==========



prefetch to device true - with both data sets slower, with just ds_train same speed, so no point
python train.py --plan=config/cnn_101_benchmark.toml --log_dir=/tmp/foo  418.21s user 233.41s system 466% cpu 2:19.64 total

prefetch to device false
python train.py --plan=config/cnn_101_benchmark.toml --log_dir=/tmp/foo  132.69s user 69.65s system 192% cpu 1:44.95 total



--> seems to have no effect


2023-05-02
==========

Simple benchmark is 3x faster on GPU (wallclock)
but maybe GPU is better again since it uses fewer CPU cores.

2023-05-01
==========

nice -19 time python generate_cnn_training_data.py "--pgn=../ChessData/Twic/twic*.pgn" --out=data/f1000.rio --shards=100
...
Open: 1247/1248: ../ChessData/Twic/twic1278.pgn : 68694.3 : 55.0
3202048 263859466 210136752 53722714
3203072 263947751 210208649 53739102
3204096 264032539 210276833 53755706
3205120 264117056 210344537 53772519
n_game:  3206053
n_move:  264199625
n_dup:  53789318
n_gen:  210410307o
    68791.44 real     62010.54 user       911.88 sys
...
du -shc data/f1000*
...
228M	data/f1000.rio-00099-of-00100
 22G	total

-----
read 2 shards
done:  2106435
done:  2102822

one shard has ~2000 batches @ 1k

batches @ 1k: 184,930
epochs: 1849 <-- 1 pass through training data



records(100%); 210,462,850  <-- based on 2 shards
n_gen:         210,410,307  <-- checks out
90%:           189369276    <-- training


2023-04-29
==========
Tensorflow fragment:

logging.set_verbosity('error')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
warnings.filterwarnings('ignore', category=Warning)

unclear if dev starts with 'source ~/miniconda3/bin/activate' based on
https://developer.apple.com/metal/tensorflow-plugin/

list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

tf.data.TFRecordDataset(
    filenames,
    compression_type=None,
    buffer_size=None,
    num_parallel_reads=None,
    name=None
)
