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
https://github.com/tensorflow/models/blob/master/official/nlp/optimization.py

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