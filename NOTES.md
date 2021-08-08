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