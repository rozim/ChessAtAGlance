#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"

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

#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_cat.h"

#include "absl/container/flat_hash_set.h"
#include "absl/numeric/int128.h"

#include "gflags/gflags.h"

using namespace std;
using open_spiel::Game;
using open_spiel::LoadGame;
using open_spiel::chess::ChessState;
using open_spiel::chess::Move;
using open_spiel::Action;

using tensorflow::Example;
using open_spiel::chess::ChessBoard;

DEFINE_string(stockfish_csv, "", "");
DEFINE_bool(shard_random, true, "Else shard by FEN/boad hash");

void process_stockfish_csv(const string& fn) {
  setbuf(stdout, NULL);
  FILE * f = fopen(fn.c_str(), "r");
  char line[1024];
  std::shared_ptr<const Game> game = LoadGame("chess");
  absl::flat_hash_set<absl::uint128> mega;
  Example ex;
  std::string ex_out;

  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  options.write_buffer_size = 32 * 1024 * 1024;
  options.block_size = 4 * 1024 * 1024;
  options.max_file_size = 16 * 1024 * 1024;

  std::vector<leveldb::DB*> dbs;
  for (int i = 0; i < 10; i++) {
    string file_name = absl::StrFormat("stockfish-d3-%d.leveldb", i);
    leveldb::DB* db;
    leveldb::Status status = leveldb::DB::Open(options, file_name, &db);
    dbs.push_back(db);
    assert(status.ok());
  }

  while (fgets(line, sizeof(line), f) != NULL) {
    line[strlen(line)-1] = '\0';
    char * comma = strchr(line, ',');
    assert(comma != NULL);
    *comma = '\0';;
    char * fen = &line[0];
    char * lan = comma + 1;

    const ChessState state(game, fen);
    if (state.CurrentPlayer() < 0) {
      continue; // Draw by rep?
    }
    const auto& board = state.Board();
    absl::optional<Move> maybe_move = board.ParseLANMove(lan);
    SPIEL_CHECK_TRUE(maybe_move);
    const Action action = MoveToAction(*maybe_move, state.BoardSize());
    absl::uint128 key = absl::MakeUint128(board.HashValue(), action);

    if (!mega.insert(key).second) {
      continue;
    }
    std::vector<float> v(game->ObservationTensorSize());
    state.ObservationTensor(state.CurrentPlayer(),
			    absl::MakeSpan(v));
    string action2s = state.ActionToString(state.CurrentPlayer(), action);
    ex.Clear();
    AppendFeatureValues(v, "board", &ex);
    AppendFeatureValues({action}, "label", &ex);
    AppendFeatureValues(state.LegalActions(), "legal_moves", &ex);
    AppendFeatureValues({action2s}, "san", &ex);
    AppendFeatureValues({maybe_move->ToLAN()}, "lan", &ex);
    AppendFeatureValues({board.ToFEN()}, "fen", &ex);
    AppendFeatureValues({board.Movenumber()}, "ply", &ex);

    ex_out.clear();
    ex.SerializeToString(&ex_out);

    // Mix up bits to address the unproven concern about
    // 'skey' in leveldb having patterns based on board.HashValue().
    key = absl::Hash<absl::uint128>{}(key);
    string skey = absl::StrFormat("%016llx" "%016llx",  Uint128High64(key), Uint128Low64(key));
    dbs[random() % 10]->Put(leveldb::WriteOptions(),
			    skey,
			    ex_out);
  }
  fclose(f);

  for (int i = 0; i < 10; i++) {
    delete dbs[i];
  }
}


int main(int argc, char * argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!FLAGS_stockfish_csv.empty()) {
    process_stockfish_csv(FLAGS_stockfish_csv);
    exit(0);
  }

  absl::flat_hash_set<absl::uint128> mega;
  time_t t1 = time(0L);
  printf("Begin\n");
  polyglot_init();
  srandom(time(0L));

  //google::InitGoogleLogging(argv[0]);

  std::shared_ptr<const Game> game = LoadGame("chess");

  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  options.write_buffer_size = 32 * 1024 * 1024;
  options.block_size = 4 * 1024 * 1024;
  options.max_file_size = 16 * 1024 * 1024;

  std::vector<leveldb::DB*> dbs;
  for (int i = 0; i < 10; i++) {
    string file_name = absl::StrFormat("mega-v5-%d.leveldb", i);
    leveldb::DB* db;
    leveldb::Status status = leveldb::DB::Open(options, file_name, &db);
    dbs.push_back(db);
    assert(status.ok());
  }

  long games = -1;
  long dups = 0;
  long long approx_bytes = 0;

  while (*++argv != NULL) {
    printf("Open %s | %ld (s)\n", *argv, time(0L) - t1);
    pgn_t pgn;
    pgn_open(&pgn, *argv);

    Example ex;
    string ex_out;
    long mod = 1;
    while (pgn_next_game(&pgn)) {
      games++;
      if (games % mod == 0) {
	time_t el = time(0L) - t1;
	if (el == 0) {
	  el = 1;
	}
	printf("game: %ld [%s | %s] | write=%ld dup=%ld | %ld (s), %ld (gps)\n", games, pgn.white, pgn.black,
	       mega.size(),
	       dups,
	       el, games/el);

	mod *= 2;
	if (mod > 5000) { mod = 5000; }
      }

      ChessState state(game);

      board_t board;
      board_start(&board);
      char str[256];
      int ply = -1;

      while (pgn_next_move(&pgn, str, 256)) {
	ply++;
	if (state.CurrentPlayer() < 0) { // maybe draw by rep recognized by spiel
	  continue; // read thru moves
	}

        int move = move_from_san(str, &board);
        if (move == MoveNone || !move_is_legal(move, &board)) {
          printf("illegal move \"%s\" at line %d, column %d\n",
                   str, pgn.move_line,pgn.move_column);
	  abort();
        }

	absl::optional<Move> maybe_move = state.Board().ParseSANMove(str);
	SPIEL_CHECK_TRUE(maybe_move);
	Action action = MoveToAction(*maybe_move, state.BoardSize());

	absl::uint128 key = absl::MakeUint128(state.Board().HashValue(), action);

	if (mega.insert(key).second) {
	  ex.Clear();
	  std::vector<float> v(game->ObservationTensorSize());
	  state.ObservationTensor(state.CurrentPlayer(),
				  absl::MakeSpan(v));
	  string action2s = state.ActionToString(state.CurrentPlayer(), action);
	  AppendFeatureValues(v, "board", &ex);
	  AppendFeatureValues({action}, "label", &ex);
	  AppendFeatureValues(state.LegalActions(), "legal_moves", &ex);
	  AppendFeatureValues({action2s}, "san", &ex);
	  AppendFeatureValues({maybe_move->ToLAN()}, "lan", &ex);
	  AppendFeatureValues({state.Board().ToFEN()}, "fen", &ex);
	  AppendFeatureValues({ply}, "ply", &ex);
	  approx_bytes += v.size() + 4 + 4 + 4 + 4 + 32;

	  ex_out.clear();
	  ex.SerializeToString(&ex_out);

	  // Mix up bits to address the unproven concern about
	  // 'skey' in leveldb having patterns based on state.Board().HashValue().
	  key = absl::Hash<absl::uint128>{}(key);
	  string skey = absl::StrFormat("%016llx" "%016llx",  Uint128High64(key), Uint128Low64(key));
	  int shard = FLAGS_shard_random ?
	    random() % 10 :
	    state.Board().HashValue() % 10; // Group same FEN into same shard.
	  dbs[shard]->Put(leveldb::WriteOptions(),
			  skey,
			  ex_out);
	} else {
	  dups++;
	}

	state.ApplyAction(action);
        move_do(&board, move);
      }
    }
    pgn_close(&pgn);
  }

  printf("All done, games=%ld\n", games);
  for (int i = 0; i < 10; i++) {
    printf("Close %d | %ld (s)\n", i, time(0L) - t1);
    delete dbs[i];
  }

  printf("Approx bytes: %lld\n", approx_bytes);
  printf("All done after %ld\n", time(0L) - t1);
}
