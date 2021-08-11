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