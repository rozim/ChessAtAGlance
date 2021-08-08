#include <stdio.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "polyglot_lib.h"

#include "open_spiel/games/chess.h"
#include "open_spiel/games/chess/chess_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"

using namespace std;
using open_spiel::Game;
using open_spiel::LoadGame;
using open_spiel::chess::ChessState;

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
  ChessState initial_state(game);
  auto shape = game->ObservationTensorShape();
  std::vector<float> v(game->ObservationTensorSize());
  initial_state.ObservationTensor(initial_state.CurrentPlayer(),
                                  absl::MakeSpan(v));  

  int games = 0;
  while (*++argv != NULL) {
    printf("Open %s\n", *argv);
    pgn_t pgn;
    pgn_open(&pgn, *argv);

    while (pgn_next_game(&pgn)) {
      board_t board;      
      board_start(&board);
      char str[256];
      while (pgn_next_move(&pgn, str, 256)) {
        int move = move_from_san(str, &board);
        if (move == MoveNone || !move_is_legal(move, &board)) {
          printf("illegal move \"%s\" at line %d, column %d\n",
                   str, pgn.move_line,pgn.move_column);	  
	  break;
        }
        move_do(&board, move);
      }
      games++;
    }
    pgn_close(&pgn);    
  }
  
  printf("All done, games=%d\n", games);
}
