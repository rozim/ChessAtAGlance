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
n_gen:  210410307
    68791.44 real     62010.54 user       911.88 sys
...
du -shc data/f1000*
...
228M	data/f1000.rio-00099-of-00100
 22G	total

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
