
2023-05-29
==========
nice -19 python -u generate_transformer_training_data.py  "--pgn=../ChessData/Twic/twic*.pgn"  --sqlite_out=data3/mega.db >& tx.shh

2023-05-15
==========

date; cat *-dump-fix.sql | python transactions-10k.py | nice -19 time sqlite3 mega.sqlite ; date
Sun May 14 13:57:30 PDT 2023
    50514.66 real      1405.38 user     21534.95 sys
Mon May 15 03:59:25 PDT 2023

ls -l -h mega.sqlite
-rw-r--r--  1 dave  staff    71G May 15 03:59 mega.sqlite

avg 333 bytes/position

sqlite3 mega.sqlite
SQLite version 3.41.1 2023-03-10 12:13:52
Enter ".help" for usage hints.
sqlite> .timer on
sqlite> select count(*) from unnamed;
228623421
Run Time: real 192.382 user 0.655760 sys 17.968960
sqlite> select rowid from unnamed limit 1;
109583245
Run Time: real 0.004 user 0.000270 sys 0.000956
sqlite> select rowid from unnamed order by random() limit 1;
109275462
Run Time: real 202.962 user 10.922697 sys 18.163202
sqlite>


nice time python sqlite3_to_recordio.py
...
QUERY:  SELECT value FROM unnamed WHERE CAST(ABS(key + rowid) AS INTEGER) % 100 == 14 ORDER BY RANDOM()
226338386 196
227678283 60

ALL DONE
    25678.29 real     12799.71 user      2366.27 sys

wc -c mega-v2.rio-000*
...
 263546446 mega-v2.rio-00098-of-00100
 263702914 mega-v2.rio-00099-of-00100
 26374343742 total
wc -c f1000.rio-000* | tail
...
 234451948 f1000.rio-00091-of-00100
 234234813 f1000.rio-00092-of-00100
 234309806 f1000.rio-00093-of-00100
 234385515 f1000.rio-00094-of-00100
 234401431 f1000.rio-00095-of-00100
 234216630 f1000.rio-00096-of-00100
 234043649 f1000.rio-00097-of-00100
 234195697 f1000.rio-00098-of-00100
 233879858 f1000.rio-00099-of-00100
 23423143484 total
(base) dave@macbook-pro data %

2023-05-14
==========


CREATE TABLE IF NOT EXISTS "unnamed" (key TEXT PRIMARY KEY, value BLOB);

2023-05-14
==========

rows in *fix.sql files, before trying to build mega db

228,623,421

2023-05-08
==========

lr tuning
strong preference for
initial lr:       0.0050000
max decay factor: 0.10000

2023-05-07
==========

test_summary_writer = tf.summary.create_file_writer('test/logdir')
with test_summary_writer.as_default():


2023-05-06
==========

Activation results - see tune/*.csv

elu 5 of top 6.
silu high.
linear surprisingly good
selu not sampled enough
gelu worse than elu
softmax, sigmoid, hard_sigmoid, log_softmax are disasters

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
