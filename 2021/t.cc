#include <stdio.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "polyglot_lib.h"

#include "open_spiel/games/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/example/feature.pb.h"
#include "tensorflow/core/example/feature_util.h"

#include "leveldb/db.h"
#include "leveldb/status.h"

#include "absl/strings/str_format.h"

using namespace std;
using open_spiel::Game;
using open_spiel::LoadGame;
using open_spiel::chess::ChessState;
using open_spiel::chess::Move;
using open_spiel::Action;

using tensorflow::Example;


int main(int argc, char * argv[]) {
  printf("Begin\n");
  polyglot_init();


  //gflags::ParseCommandLineFlags(&argc, &argv, true);  
  //google::InitGoogleLogging(argv[0]);

  /*
  open_spiel::GameParameters params;
  std::shared_ptr<const open_spiel::Game> game =
    open_spiel::LoadGame("chess", params);

  if (!game) {
    std::cerr << "problem with loading game, exiting..." << std::endl;
    return -1;
  }
  std::unique_ptr<open_spiel::State> state = game->NewInitialState();



  std::cerr << "Initial state:" << std::endl;
  std::cerr << "State:" << std::endl << state->ToString() << std::endl;
  */
  std::shared_ptr<const Game> game = LoadGame("chess");
  {
    ChessState state(game);
    auto shape = game->ObservationTensorShape();
    std::vector<float> v(game->ObservationTensorSize());
    state.ObservationTensor(state.CurrentPlayer(),
			    absl::MakeSpan(v));
    printf("v=%d\n", (int) v.size());

    Example ex;
    ex.Clear();
    AppendFeatureValues(v, "board", &ex);
    printf("YES: %s\n", ex.DebugString().c_str());

    string ex_out;
    ex.SerializeToString(&ex_out);

    leveldb::Options options;
    options.create_if_missing = true;
    string file_name = "foo.leveldb";
    leveldb::DB* db;
    leveldb::Status status = leveldb::DB::Open(options, absl::StrFormat("tmp-%s", file_name.c_str()), &db);
    assert(status.ok());

    status = db->Put(leveldb::WriteOptions(), "a", ex_out);
    assert(status.ok());

    delete db;
  }

  int games = 0;
  while (*++argv != NULL) {
    printf("Open %s\n", *argv);
    pgn_t pgn;
    pgn_open(&pgn, *argv);

    while (pgn_next_game(&pgn)) {
      printf("\n");
      ChessState state(game);
      
      board_t board;      
      board_start(&board);
      char str[256];
      while (pgn_next_move(&pgn, str, 256)) {
        int move = move_from_san(str, &board);
        if (move == MoveNone || !move_is_legal(move, &board)) {
          printf("illegal move \"%s\" at line %d, column %d\n",
                   str, pgn.move_line,pgn.move_column);	  
	  abort();
        }


	absl::optional<Move> maybe_move = state.Board().ParseSANMove(str);
	SPIEL_CHECK_TRUE(maybe_move);
	Action action = MoveToAction(*maybe_move, state.BoardSize());
	string foo = state.ActionToString(state.CurrentPlayer(), action);
	state.ApplyAction(action);
	//SPIEL_CHECK_EQ(state.Board().ToFEN(), fen_after);
	//state.UndoAction(player, action);
	//SPIEL_CHECK_EQ(state.Board().ToFEN(), fen);

	printf("SAN: %s : %s : %s : %lld\n", str, state.Board().ToFEN().c_str(), foo.c_str(), action);

        move_do(&board, move);
      }
      games++;
    }
    pgn_close(&pgn);    
  }
  
  printf("All done, games=%d\n", games);
}
