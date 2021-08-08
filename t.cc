#include <stdio.h>

#include "polyglot_lib.h"

using namespace std;

int main(int argc, char * argv[]) {
  printf("Begin\n");
  //gflags::ParseCommandLineFlags(&argc, &argv, true);  
  //google::InitGoogleLogging(argv[0]);
  polyglot_init();
  int games = 0;
  while (*++argv != NULL) {
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
