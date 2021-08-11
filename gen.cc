#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

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
//#include "absl/random/random.h"

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
  srandom(time(0L));

  //gflags::ParseCommandLineFlags(&argc, &argv, true);  
  //google::InitGoogleLogging(argv[0]);

  //absl::BitGen gen;
//
//   // Generate an integer value in the closed interval [1,6]
//   int die_roll = absl::uniform_int_distribution<int>(1, 6)(gen);
  std::shared_ptr<const Game> game = LoadGame("chess");

  leveldb::Options options;
  options.create_if_missing = true;
  string file_name = "gen.leveldb";
  leveldb::DB* db;
  leveldb::Status status = leveldb::DB::Open(options, file_name, &db);
  assert(status.ok());

  //status = db->Put(leveldb::WriteOptions(), "a", ex_out);
  //assert(status.ok());

  int games = 0;
  long long approx_bytes = 0;
  
  while (*++argv != NULL) {
    printf("Open %s\n", *argv);
    pgn_t pgn;
    pgn_open(&pgn, *argv);

    Example ex;
    string ex_out;

    while (pgn_next_game(&pgn)) {
      printf("\n");
      ChessState state(game);
      
      board_t board;      
      board_start(&board);
      char str[256];

      while (pgn_next_move(&pgn, str, 256)) {
	ex.Clear();	
        int move = move_from_san(str, &board);
        if (move == MoveNone || !move_is_legal(move, &board)) {
          printf("illegal move \"%s\" at line %d, column %d\n",
                   str, pgn.move_line,pgn.move_column);	  
	  abort();
        }

	absl::optional<Move> maybe_move = state.Board().ParseSANMove(str);
	SPIEL_CHECK_TRUE(maybe_move);
	Action action = MoveToAction(*maybe_move, state.BoardSize());

	std::vector<float> v(game->ObservationTensorSize());
	state.ObservationTensor(state.CurrentPlayer(),
				absl::MakeSpan(v));
	string action2s = state.ActionToString(state.CurrentPlayer(), action);
	AppendFeatureValues(v, "board", &ex);
	AppendFeatureValues(absl::StrFormat("%ldd", action), "label", &ex); 	
	AppendFeatureValues(action2s, "action", &ex); 
	AppendFeatureValues(state.Board().ToFEN(), "fen", &ex);

	approx_bytes += v.size() + 4 + 4 + 4 + 32;
	
	ex_out.clear();
	ex.SerializeToString(&ex_out);
	db->Put(leveldb::WriteOptions(),
		//absl::StrFormat("%d", absl::uniform_int_distribution<int>()(gen)),
		absl::StrFormat("%ld", random()),
		ex_out);

	state.ApplyAction(action);
	//SPIEL_CHECK_EQ(state.Board().ToFEN(), fen_after);
	//state.UndoAction(player, action);
	//SPIEL_CHECK_EQ(state.Board().ToFEN(), fen);

	//printf("SAN: %s : %s : %s : %lld\n", str, state.Board().ToFEN().c_str(), foo.c_str(), action);

        move_do(&board, move);
      }
      games++;
    }
    pgn_close(&pgn);    
  }
  
  printf("All done, games=%d\n", games);
  delete db;
  printf("All done");
  printf("Approx bytes: %lld\n", approx_bytes);
}
