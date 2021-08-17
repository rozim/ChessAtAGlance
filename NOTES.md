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
