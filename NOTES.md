# Open Spiel (chess)

## from open_spiel/games/chess/chess_board.h
// Forward declare ChessBoard here because it's needed in Move::ToSAN.
class ChessBoard;

class Move
	ToString()
	ToLAN()
	ToSan()

kDefaultStandardFEN

class ChessBoard
	BoardFromFEN()
	GenerateLegalMoves
	ParseMove
	ParseSANMove
	ParseLANMove
	ApplyMove
	ToFEN()
	ChessBoard MakeDefaultBoard();
	std::string DefaultFen(int board_size);

## from open_spiel/games/chess.h
NumDistinctActions()
ObservationTensorShape
Action MoveToAction(const Move& move, int board_size = kDefaultBoardSize);

Move ActionToMove(const Action& action, const ChessBoard& board);

class ChessState
  std::vector<Action> LegalActions() const override;
    void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
    ChessBoard& Board() { return current_board_; }
    const ChessBoard& Board() const { return current_board_; }
    ChessBoard& StartBoard() { return start_board_; }
    Action ParseMoveToAction(const std::string& move_str) const;

class ChessGame : public Game {


## Tensorflow

### RecordWriter compression
bool IsZlibCompressed(const RecordWriterOptions& options) {
  return options.compression_type == RecordWriterOptions::ZLIB_COMPRESSION;
}

bool IsSnappyCompressed(const RecordWriterOptions& options) {
  return options.compression_type == RecordWriterOptions::SNAPPY_COMPRESSION;
}
...

      io::RecordWriterOptions options;
      options.compression_type = io::RecordWriterOptions::SNAPPY_COMPRESSION;
      options.zlib_options.output_buffer_size = buf_size;

# Building tensorflow dev
/Users/dave/Projects/tensorflow
bazel build  //tensorflow/tools/pip_package:build_pip_package

/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/bin/python3.8
/System/Volumes/Data/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.8/bin/python3.8

bazel build  //tensorflow/core/lib/io:record_writer
  bazel-bin/tensorflow/core/lib/io/librecord_writer.lo
  bazel-bin/tensorflow/core/lib/io/librecord_writer.pic.lo
  bazel-bin/tensorflow/core/lib/io/librecord_writer.so

maybe need bazel build  --config=opt

huge hack:
LD_LIBRARY_PATH=/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow ./t t1.pgn

Open Spiel + Tensorflow C++ issues
https://github.com/deepmind/open_spiel/issues/172
https://github.com/deepmind/open_spiel/pull/307

riegeli: PITA, may be obsolete?

-----

g++ -std=c++17 -L/usr/local/lib -lleveldb t2.cc -o t2

-----
LD_LIBRARY_PATH=/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow ./gen t1.pgn

-----
ex = tf.train.Example().FromString(ent[1])

-----
inline constexpr int NumDistinctActions() { return 4672; }
inline const std::vector<int>& ObservationTensorShape() {
  static std::vector<int> shape{
      13 /* piece types * colours + empty */ + 1 /* repetition count */ +
          1 /* side to move */ + 1 /* irreversible move counter */ +
          4 /* castling rights */,
      kMaxBoardSize, kMaxBoardSize};
  return shape;
}
20 * 64 = 1280

feature_spec = { 'board': tf.io.FixedLenFeature([1280], tf.float32)}

ent = next(db.RangeIter())
ex = tf.train.Example().FromString(ent[1])
foo = ex.features.feature['board'].float_list.value
tf.convert_to_tensor(foo)

------------------------------------------------------------
2021-08-13
dense(4096) gets 0.17 accuracy
256/256 [==============================] - 16s 64ms/step - loss: 3.3211 - accuracy: 0.1748 - val_loss: 3.3289 - val_accuracy: 0.1740


------------------------------------------------------------
2021-08-15
conda activate tf25

------------------------------------------------------------
2021-08-16
iterating on all data seems to go on forever
...
4194304 1265
8388608 2537

------------------------------------------------------------
{LD_LIBRARY_PATH}:/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow

level_db_read, ~75x as fast as python
...
8388608 34
16777216 71

16925485 72

rerun with sizes
16925485 75 (s) 89604555283 (bytes)
83G uncompressed
9G compressed
16,925,485 --> 16M rows
10x files
170M rows

16925485 / 128 (batch size)
132230 batches


132230 / 256 (steps per epoch)
516 epochs --> 500 epochs should cover 1 file (once)

see this later
https://lczero.org/dev/wiki/technical-explanation-of-leela-chess-zero/

warmup
bhttps://github.com/tensorflow/models/blob/master/official/nlp/optimization.py

adamw
https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/weight_decay_optimizers.py


results/2021-08-17_09:44:22
val_accuracy    0.2258 (best)
                0.2230 (last)
test_accuracy : 0.2154 (last)

val_accuracy    0.2309 (best)
                0.2207 (last)
test_accuracy : 0.2189 (last)

------------------------------------------------------------
bs=1024 works best in an evaluation test -2048 is same, and 512 is slower, normalized for same # examples

------------------------------------------------------------
FileOutputStream
ZeroCopyOutputStream example
GzipOutputStream
wire_format_lite
WriteInt32NoTag
Message:SerializeToZeroCopyStream
-----
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow

------------------------------------------------------------
eof crash, but hit 0.3176

poch 166/200
 17/100 [====>.........................] - ETA: 36s - loss: 2.3463 - accuracy: 0.3176 Traceback (most recent call last):
  File "/Users/dave/Projects/ChessAtAGlance/train.py", line 187, in <module>

  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/absl/app.py", line 300, in run
    _run_main(main, args)
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/absl/app.py", line 251, in _run_main
    sys.exit(main(argv))
  File "/Users/dave/Projects/ChessAtAGlance/train.py", line 131, in main
    filepath=best_path,
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/keras/engine/training.py", line 1183, in fit
    tmp_logs = self.train_function(iterator)
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 889, in __call__
    result = self._call(*args, **kwds)
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/eager/def_function.py", line 917, in _call
    return self._stateless_fn(*args, **kwds)  # pylint: disable=not-callable
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/eager/function.py", line 3023, in __call__
    return graph_function._call_flat(
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/eager/function.py", line 1960, in _call_flat
    return self._build_call_outputs(self._inference_function.call(
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/eager/function.py", line 591, in call
    outputs = execute.execute(
  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/eager/execute.py", line 59, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.UnknownError: 2 root error(s) found.
  (0) Unknown:  error: unpack requires a buffer of 4 bytes
Traceback (most recent call last):

  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/ops/script_ops.py", line 249, in __call__
    ret = func(*args)

  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py", line 645, in wrapper
    return func(*args, **kwargs)

  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/data/ops/dataset_ops.py", line 961, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))

  File "/Users/dave/Projects/ChessAtAGlance/data.py", line 19, in gen_snappy
    for ex in unsnappy(fn):

  File "/Users/dave/Projects/ChessAtAGlance/snappy_io.py", line 16, in unsnappy
    n = unpack('@i', read(4))[0]

struct.error: unpack requires a buffer of 4 bytes


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
	 [[ArgMax/_26]]
  (1) Unknown:  error: unpack requires a buffer of 4 bytes
Traceback (most recent call last):

  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/ops/script_ops.py", line 249, in __call__
    ret = func(*args)

  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/autograph/impl/api.py", line 645, in wrapper
    return func(*args, **kwargs)

  File "/Users/dave/miniforge3/envs/tf25/lib/python3.9/site-packages/tensorflow/python/data/ops/dataset_ops.py", line 961, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))

  File "/Users/dave/Projects/ChessAtAGlance/data.py", line 19, in gen_snappy
    for ex in unsnappy(fn):

  File "/Users/dave/Projects/ChessAtAGlance/snappy_io.py", line 16, in unsnappy
    n = unpack('@i', read(4))[0]

struct.error: unpack requires a buffer of 4 bytes


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_train_function_2312]

Function call stack:
train_function -> train_function
------------------------------------------------------------
steps * bs * epochs (when bug above happened)
100 * 1024 * 166
     16998400 <-- approx
     16896000 <-- mul * 165, closer
done 16922368 <-- read all

------------------------------------------------------------
2021-08-20
v13.toml
100/100 [==============================] - 105s 1s/step - loss: 2.0492 - accuracy: 0.3752 - val_loss: 2.0779 - val_accuracy: 0.3646
Write results/2021-08-19_22:56:51/last.model
Test (last)
256/256 [==============================] - 189s 738ms/step - loss: 2.0631 - accuracy: 0.3718
Test: {'loss': 2.0630764961242676, 'accuracy': 0.37175750732421875} 188
Test (best)
Open mega-v2-2.snappy
256/256 [==============================] - 194s 757ms/step - loss: 2.0631 - accuracy: 0.3718
Test/2: {'loss': 2.0630764961242676, 'accuracy': 0.37175750732421875} 194
Write results/2021-08-19_22:56:51/history.csv
Write results/2021-08-19_22:56:51/report.txt
val_accuracy    0.3661 (best)
                0.3646 (last)
test_accuracy   0.3718 (best)
                0.3718 (last)

-----
Epoch 200/200
100/100 [==============================] - 124s 1s/step - loss: 1.9918 - accuracy: 0.3850 - val_loss: 2.0192 - val_accuracy: 0.3794
Write results/2021-08-20_11:55:04/last.model
Test (last)
256/256 [==============================] - 202s 790ms/step - loss: 2.0019 - accuracy: 0.3847
Test: {'loss': 2.0018813610076904, 'accuracy': 0.38466644287109375} 202
Test (best)
Open mega-v2-2.snappy
256/256 [==============================] - 205s 798ms/step - loss: 2.0019 - accuracy: 0.3847
Test/2: {'loss': 2.0018813610076904, 'accuracy': 0.38466644287109375} 205
Write results/2021-08-20_11:55:04/history.csv
Write results/2021-08-20_11:55:04/report.txt
val_accuracy    0.3798 (best)
                0.3794 (last)
test_accuracy   0.3847 (best)
                0.3847 (last)
python train.py --plan=v14.toml  10599.02s user 7603.45s system 71% cpu 7:05:41.52 total
(tf25) dave@daves-air ChessAtAGlance %
------------------------------------------------------------

Epoch 200/200
100/100 [==============================] - 155s 2s/step - loss: 1.9854 - accuracy: 0.3870 - val_loss: 2.0118 - val_accuracy: 0.3799
Write results/2021-08-20_23:56:36/last.model
Test (last)
256/256 [==============================] - 220s 860ms/step - loss: 1.9956 - accuracy: 0.3851
Test: {'loss': 1.9955915212631226, 'accuracy': 0.38512420654296875} 220
Test (best)
Open mega-v2-2.snappy
256/256 [==============================] - 222s 868ms/step - loss: 1.9956 - accuracy: 0.3851
Test/2: {'loss': 1.9955915212631226, 'accuracy': 0.38512420654296875} 224
Write results/2021-08-20_23:56:36/history.csv
Write results/2021-08-20_23:56:36/report.txt
val_accuracy    0.3806 (best)
                0.3799 (last)
test_accuracy   0.3851 (best)
                0.3851 (last)
python train.py --plan=v15.toml  11724.31s user 9607.58s system 65% cpu 9:02:50.86 total

------------------------------------------------------------
2021-08-23

#def wrap_generator(filename):
#  return tf.data.Dataset.from_generator(parse_file(filename), [tf.int32, tf.int32])

# files = tf.data.Dataset.from_tensor_slices(files_to_process)
# dataset = files.apply(tf.contrib.data.parallel_interleave(wrap_generator, cycle_length=N))
# dataset = dataset.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
# dataset = dataset.shuffle(shuffle_size).batch(batch_size).prefetch(2)
# it = dataset.make_one_shot_iterator()
# https://stackoverflow.com/questions/52179857/parallelize-tf-from-generator-using-tf-contrib-data-parallel-interleave

TBD: tf.py_function?

------------------------------------------------------------
2021-08-24

disaster, something went wrong, val loss huge, can't narrow down
with git bisect

git bisect good 7d4e0beedc0a459597311528cb9e45b380b097c8

------------------------------------------------------------
2021-08-25

TFRecord format seems 60x as fast in benchmark.
Wow.

------------------------------------------------------------
2021-08-26
==========

shard by FEN
LD_LIBRARY_PATH=/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow ./gen --shard_random=false ~/Projects/ChessData/mega2400_part_*.pgn
...
game: 2810000 [Zombirt,F | Brzezinski,M] | write=169019751 dup=57587073 | 19791 (s), 141 (gps)
All done, games=2813613
...
Close 9 | 19836 (s)
Approx bytes: 224740380192
All done after 19836

2021-08-27
==========
python gen_stockfish.py -d 1 -n 1048576 > stockfish-v5-d1.csv &
python gen_stockfish.py -d 3 -n 1048576 > stockfish-v5-d3.csv &

rm -rf stockfish-v5-d1-?.leveldb ; LD_LIBRARY_PATH=/Users/dave/Projects/open_spiel/build:/Users/dave/miniforge3/lib/python3.9/site-packages/tensorflow  ./gen  --shard_random=False --stockfish_csv=stockfish_d1_v5_1M.csv


2021-08-28
==========

- prefetch_to_device=false

Epoch 25/25
100/100 [==============================] - 9s 89ms/step - loss: 2.2342 - accuracy: 0.3502 - val_loss: 8.0321 - val_accuracy: 0.0463
...
Write results/2021-08-28_11:21:54/history.csv
Write results/2021-08-28_11:21:54/report.txt
val_accuracy    0.0503 (best)
                0.0463 (last)
test_accuracy   0.0459 (best)
                0.0459 (last)
Timing
on_train_batch |     2500 | 153.53
on_test_batch |     2500 | 76.01
on_test      |       25 | 76.37
on_epoch     |       25 | 267.15
on_train     |        1 | 267.19

- prefetch_to_device=true, prefetch=4

Epoch 25/25
100/100 [==============================] - 9s 88ms/step - loss: 2.0242 - accuracy: 0.3835 - val_loss: 7.6495 - val_accuracy: 0.0447
Write results/2021-08-28_11:31:20/last.model
...
Write results/2021-08-28_11:31:20/history.csv
Write results/2021-08-28_11:31:20/report.txt
val_accuracy    0.0542 (best)
                0.0447 (last)
test_accuracy   0.0454 (best)
                0.0454 (last)
Timing
on_train_batch |     2500 | 146.21
on_test_batch |     2500 | 69.96
on_test      |       25 | 70.23
on_epoch     |       25 | 251.79
on_train     |        1 | 251.84

--> earlier epochs are a few ms faster however

- prefetch_to_device=true, prefetch=1 (slower)

on_train_batch |     2500 | 153.80
on_test_batch |     2500 | 76.05
on_test      |       25 | 76.37
on_epoch     |       25 | 268.78
on_train     |        1 | 268.84

- prefetch_to_device=true, prefetch=4 (repeat from above)

Timing
on_train_batch |     2500 | 148.26
on_test_batch |     2500 | 71.08
on_test      |       25 | 71.37
on_epoch     |       25 | 256.56
on_train     |        1 | 256.61

--> somewhat similar

- prefetch_to_device=true, prefetch=2

Timing
on_train_batch |     2500 | 147.84
on_test_batch |     2500 | 70.35
on_test      |       25 | 70.62
on_epoch     |       25 | 254.86
on_train     |        1 | 254.91


- prefetch_to_device=false, prefetch=0
on_train_batch |     2500 | 150.44
on_test_batch |     2500 | 74.08
on_test      |       25 | 74.39
on_epoch     |       25 | 260.04
on_train     |        1 | 260.11

- prefetch_to_device=false, prefetch=1
Timing
on_train_batch |     2500 | 148.25
on_test_batch |     2500 | 71.02
on_test      |       25 | 71.32
on_epoch     |       25 | 254.87
on_train     |        1 | 254.93

- prefetch_to_device=true, prefetch=1
on_train_batch   |     2500 | 148.96
on_test_batch    |     2500 | 70.94
on_test          |       25 | 71.20
on_epoch         |       25 | 256.52
on_train         |        1 | 256.58

- prefetch_to_device=true, prefetch=0, buffer_size=16
on_train_batch   |     2500 | 144.93
on_test_batch    |     2500 | 70.09
on_test          |       25 | 70.36
on_epoch         |       25 | 250.98
on_train         |        1 | 251.05

- prefetch_to_device=true, prefetch=0, buffer_size=16
on_train_batch   |     2500 | 147.39
on_test_batch    |     2500 | 70.51
on_test          |       25 | 70.79
on_epoch         |       25 | 254.63
on_train         |        1 | 254.69

- prefetch_to_device=true, prefetch=0, buffer_size=16
on_train_batch   |     2500 | 148.48
on_test_batch    |     2500 | 71.93
on_test          |       25 | 72.22
on_epoch         |       25 | 258.89
on_train         |        1 | 258.94

- prefetch_to_device=true, prefetch=0, buffer_size=2
Timing
on_train_batch   |     2500 | 149.65
on_test_batch    |     2500 | 72.47
on_test          |       25 | 72.77
on_epoch         |       25 | 259.98
on_train         |        1 | 260.03

- prefetch_to_device=true, prefetch=0, buffer_size=8
Timing
on_train_batch   |     2500 | 152.59
on_test_batch    |     2500 | 74.56
on_test          |       25 | 74.88
on_epoch         |       25 | 266.80
on_train         |        1 | 266.85

- prefetch_to_device=true, prefetch=0, buffer_size=None horrible
on_train_batch   |     2500 | 163.93
on_test_batch    |     2500 | 85.79
on_test          |       25 | 86.20
on_epoch         |       25 | 293.91
on_train         |        1 | 293.96

- prefetch_to_device=true, prefetch=0, buffer_size=32 bad
on_train_batch   |     2500 | 160.99
on_test_batch    |     2500 | 84.21
on_test          |       25 | 84.58
on_epoch         |       25 | 287.98
on_train         |        1 | 288.03

- ugh, change methodology
- device=true, prefetch=0, buffer=2
on_train_batch   |     1000 | 59.41

- device=true, prefetch=0, buffer=2, swap=true <-- no change
on_train_batch   |     1000 | 59.03

- device=true, prefetch=2, buffer=2, swap=false <- faster slightly
on_train_batch   |     1000 | 58.64

- device=true, prefetch=2, buffer=4, swap=false
on_train_batch   |     1000 | 59.83

- device=false, prefetch=2
on_train_batch   |     1000 | 59.98

- device=false, prefetch=0
on_train_batch   |     1000 | 59.76

- device=false, prefetch=0

- device=false, prefetch=4
on_train_batch   |     1000 | 63.02

- device=false, prefetch=1
on_train_batch   |     1000 | 65.67

- device=true buffer=2 prefetch=1
on_train_batch   |     1000 | 68.12

- device=false prefetch=2
on_train_batch   |     1000 | 65.82

- device=false prefetch=0
on_train_batch   |     1000 | 59.89

- longer, prefetch=0
on_train_batch   |     6250 | 366.84

- longer, prefetch=1
on_train_batch   |     6250 | 365.77

- longer, prefetch=8
on_train_batch   |     6250 | 365.77

- longer, prefetch=8? unclear what changed
on_train_batch   |     6250 | 379.46


2021-08-29
==========

var len legal move list (SparseTensor) to multi one hot

dense = tf.sparse.to_dense(v, default_value=-1)
hot = tf.one_hot(dense, on_value=1.0, off_value=0.0, depth=NUM_CLASSES)
hot2 = tf.math.reduce_sum(hot, axis=[-2])

Santi paper
https://arxiv.org/pdf/2006.14171.pdf

2021-08-30
==========

Prelim mask - 430 ms/step to 555ms/step - so masking in model is slow, but doesn't seem insanely slow
vs doing it in data seemed to slow things down more

2021-08-30
==========

Epoch 99/100
100/100 [==============================] - 56s 558ms/step - loss: 2.2159 - accuracy: 0.3322 - val_loss: 2.6037 - val_accuracy: 0.2486
Epoch 100/100
100/100 [==============================] - 55s 552ms/step - loss: 2.2091 - accuracy: 0.3342 - val_loss: 2.5551 - val_accuracy: 0.2562
Write results/2021-08-30_10:05:52/last.model
Test (last)
2048/2048 [==============================] - 354s 173ms/step - loss: 2.2627 - accuracy: 0.3211
Test: {'loss': 2.262695074081421, 'accuracy': 0.32108020782470703} 353
Test (best)
Open ['mega-v5-9.recordio', 'stockfish-v5-d1-9.recordio', 'stockfish-v5-d3-9.recordio']
2048/2048 [==============================] - 358s 175ms/step - loss: 2.2627 - accuracy: 0.3211
Test/2: {'loss': 2.262695074081421, 'accuracy': 0.32108020782470703} 359
Write results/2021-08-30_10:05:52/history.csv
Write results/2021-08-30_10:05:52/report.txt
val_accuracy    0.2589 (best)
                0.2562 (last)
test_accuracy   0.3211 (best)
                0.3211 (last)
Timing
on_train_batch   |    10000 | 5074.57
on_test_batch    |     2500 | 522.51
on_test          |      100 | 525.11
on_epoch         |      100 | 5950.81
on_train         |        1 | 5951.06
on_fit           |        1 | 5951.09
after_fit        |        1 | 713.81
overall          |        1 | 6668.58
(tf25) dave@daves-air ChessAtAGlance % !!:p
# python train.py --plan=v17-mask.toml


2021-08-30
==========

fewer files, take out stockfish

# python train.py --plan=v17-mask2.toml
Epoch 100/100
100/100 [==============================] - 55s 552ms/step - loss: 2.1492 - accuracy: 0.3455 - val_loss: 2.1654 - val_accuracy: 0.3439
Write results/2021-08-30_12:23:46/last.model
Test (last)
2048/2048 [==============================] - 349s 171ms/step - loss: 2.1589 - accuracy: 0.3455
Test: {'loss': 2.1588590145111084, 'accuracy': 0.3454766273498535} 349
Test (best)
Open ['mega-v5-9.recordio']
2048/2048 [==============================] - 356s 174ms/step - loss: 2.1589 - accuracy: 0.3455
Test/2: {'loss': 2.1588590145111084, 'accuracy': 0.3454766273498535} 357
Write results/2021-08-30_12:23:46/history.csv
Write results/2021-08-30_12:23:46/report.txt
val_accuracy    0.3439 (best)
                0.3439 (last)
test_accuracy   0.3455 (best)
                0.3455 (last)
Timing

-----

# python train.py --plan=v17-mask3.toml

100/100 [==============================] - 112s 1s/step - loss: 2.0890 - accuracy: 0.3593 - val_loss: 2.0881 - val_accuracy: 0.3586
Epoch 100/100
100/100 [==============================] - 111s 1s/step - loss: 2.0729 - accuracy: 0.3644 - val_loss: 2.1016 - val_accuracy: 0.3575
Write results/2021-08-30_14:55:26/last.model
Test (last)
2048/2048 [==============================] - 707s 345ms/step - loss: 2.0938 - accuracy: 0.3593
Test: {'loss': 2.093775510787964, 'accuracy': 0.3592720031738281} 707
Test (best)
Open ['mega-v5-9.recordio']
2048/2048 [==============================] - 713s 348ms/step - loss: 2.0938 - accuracy: 0.3593
Test/2: {'loss': 2.093775510787964, 'accuracy': 0.3592720031738281} 715
Write results/2021-08-30_14:55:26/history.csv
Write results/2021-08-30_14:55:26/report.txt
val_accuracy    0.3587 (best)
                0.3575 (last)
test_accuracy   0.3593 (best)
                0.3593 (last)




# python train.py --plan=v17-mask4.toml
Epoch 99/100
100/100 [==============================] - 89s 893ms/step - loss: 2.2089 - accuracy: 0.3342 - val_loss: 2.2092 - val_accuracy: 0.3303
Epoch 100/100
100/100 [==============================] - 92s 926ms/step - loss: 2.1945 - accuracy: 0.3356 - val_loss: 2.2083 - val_accuracy: 0.3314
Write results/2021-08-30_18:33:04/last.model
Test (last)
2048/2048 [==============================] - 529s 258ms/step - loss: 2.2037 - accuracy: 0.3344
Test: {'loss': 2.203660726547241, 'accuracy': 0.3344287872314453} 528
Test (best)
Open ['mega-v5-9.recordio']
2048/2048 [==============================] - 523s 255ms/step - loss: 2.2037 - accuracy: 0.3344
Test/2: {'loss': 2.203660726547241, 'accuracy': 0.3344287872314453} 525
Write results/2021-08-30_18:33:04/history.csv
Write results/2021-08-30_18:33:04/report.txt
val_accuracy    0.3314 (best)
                0.3314 (last)
test_accuracy   0.3344 (best)
                0.3344 (last)
--> previous is better

2021-08-31
==========

# python train.py --plan=v18-squeeze1.toml
100/100 [==============================] - 128s 1s/step - loss: 2.1861 - accuracy: 0.3392 - val_loss: 2.2000 - val_accuracy: 0.3339
Epoch 100/100
100/100 [==============================] - 128s 1s/step - loss: 2.2205 - accuracy: 0.3306 - val_loss: 2.2425 - val_accuracy: 0.3269
Write results/2021-08-30_23:09:59/last.model
Test (last)
1024/1024 [==============================] - 374s 365ms/step - loss: 2.2407 - accuracy: 0.3281
Test: {'loss': 2.240713596343994, 'accuracy': 0.3280668258666992} 374
Test (best)
Open ['mega-v5-9.recordio']
1024/1024 [==============================] - 377s 368ms/step - loss: 2.2407 - accuracy: 0.3281
Test/2: {'loss': 2.240713596343994, 'accuracy': 0.3280668258666992} 379
Write results/2021-08-30_23:09:59/history.csv
Write results/2021-08-30_23:09:59/report.txt
val_accuracy    0.3368 (best)
                0.3269 (last)
test_accuracy   0.3281 (best)
                0.3281 (last)

--> not great

# python train.py --plan=v18-squeeze2.toml
Epoch 99/100
100/100 [==============================] - 116s 1s/step - loss: 2.1385 - accuracy: 0.3487 - val_loss: 2.1415 - val_accuracy: 0.3464
Epoch 100/100
100/100 [==============================] - 115s 1s/step - loss: 2.1256 - accuracy: 0.3518 - val_loss: 2.1369 - val_accuracy: 0.3464
Write results/2021-08-31_07:41:34/last.model
Test (last)
1024/1024 [==============================] - 345s 337ms/step - loss: 2.1354 - accuracy: 0.3488
Test: {'loss': 2.135408639907837, 'accuracy': 0.34882068634033203} 344
Test (best)
Open ['mega-v5-9.recordio']
1024/1024 [==============================] - 347s 338ms/step - loss: 2.1354 - accuracy: 0.3488
Test/2: {'loss': 2.135408639907837, 'accuracy': 0.34882068634033203} 349
Write results/2021-08-31_07:41:34/history.csv
Write results/2021-08-31_07:41:34/report.txt
val_accuracy    0.3464 (best)
                0.3464 (last)
test_accuracy   0.3488 (best)
                0.3488 (last)

[main f335904] this helps, reoreder model squeeze excite
--> reordering model a bit changes things


# python train.py --plan=v18-squeeze3.toml
Epoch 100/100
100/100 [==============================] - 117s 1s/step - loss: 2.2887 - accuracy: 0.3169 - val_loss: 2.3056 - val_accuracy: 0.3117
Write results/2021-08-31_11:59:14/last.model
Test (last)
1024/1024 [==============================] - 348s 340ms/step - loss: 2.2987 - accuracy: 0.3146
Test: {'loss': 2.2987234592437744, 'accuracy': 0.3145589828491211} 348
Test (best)
Open ['mega-v5-9.recordio']
1024/1024 [==============================] - 345s 337ms/step - loss: 2.2987 - accuracy: 0.3146
Test/2: {'loss': 2.2987234592437744, 'accuracy': 0.3145589828491211} 347
Write results/2021-08-31_11:59:14/history.csv
Write results/2021-08-31_11:59:14/report.txt
val_accuracy    0.3117 (best)
                0.3117 (last)
test_accuracy   0.3146 (best)
                0.3146 (last)
Timing
on_train_batch   |    10000 | 11017.91
on_test_batch    |     2500 | 891.22
on_test          |      100 | 894.34
on_epoch         |      100 | 12395.91
on_train         |        1 | 12396.17
on_fit           |        1 | 12396.20
after_fit        |        1 | 695.45
overall          |        1 | 13096.54

--> BAD


# python train.py --plan=v18-squeeze4.toml

Epoch 100/100
100/100 [==============================] - 146s 1s/step - loss: 2.4512 - accuracy: 0.2800 - val_loss: 2.5133 - val_accuracy: 0.2738
Write results/2021-08-31_17:33:17/last.model
Test (last)
1024/1024 [==============================] - 456s 445ms/step - loss: 2.5112 - accuracy: 0.2761
Test: {'loss': 2.5112030506134033, 'accuracy': 0.27611827850341797} 456
Test (best)
Open ['mega-v5-9.recordio']
1024/1024 [==============================] - 435s 425ms/step - loss: 2.5112 - accuracy: 0.2761
Test/2: {'loss': 2.5112030506134033, 'accuracy': 0.27611827850341797} 437
Write results/2021-08-31_17:33:17/history.csv
Write results/2021-08-31_17:33:17/report.txt
val_accuracy    0.3222 (best)
                0.2738 (last)
test_accuracy   0.2761 (best)
                0.2761 (last)

--> BAD
