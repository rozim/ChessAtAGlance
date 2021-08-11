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
#include "open_spiel/spiel_globals.h"

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
  time_t t1 = time(0L);
  printf("Begin\n");
  polyglot_init();
  srandom(time(0L));

  //gflags::ParseCommandLineFlags(&argc, &argv, true);  
  //google::InitGoogleLogging(argv[0]);

  std::shared_ptr<const Game> game = LoadGame("chess");

  leveldb::Options options;
  options.create_if_missing = true;
  string file_name = "gen.leveldb";
  leveldb::DB* db;
  leveldb::Status status = leveldb::DB::Open(options, file_name, &db);
  assert(status.ok());

  int games = -1;
  long long approx_bytes = 0;
  
  while (*++argv != NULL) {
    printf("Open %s\n", *argv);
    pgn_t pgn;
    pgn_open(&pgn, *argv);

    Example ex;
    string ex_out;
    long mod = 1;
    while (pgn_next_game(&pgn)) {
      if (games % mod == 0) {
	time_t el = time(0L) - t1;
	if (el == 0) {
	  el = 1;
	}
	printf("game: %d [%s | %s] %ld (s), %ld (gps)\n", games, pgn.white, pgn.black, el, games/el);

	mod *= 2;
      }
      games++;
      ChessState state(game);
      
      board_t board;      
      board_start(&board);
      char str[256];

      while (pgn_next_move(&pgn, str, 256)) {
	if (state.CurrentPlayer() < 0) { // maybe draw by rep recognized by spiel
	  continue; // read thru moves 
	}
	
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
		absl::StrFormat("%ld", random()),
		ex_out);

	state.ApplyAction(action);
        move_do(&board, move);
      }
    }
    pgn_close(&pgn);    
  }
  
  printf("All done, games=%d\n", games);
  delete db;

  printf("Approx bytes: %lld\n", approx_bytes);
  printf("All done after %ld\n", time(0L) - t1);
}
