// Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/chinese_chess/board.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace chinese_chess {

static bool InEnemyArea(const Point& point, Color our_color) {
  if (our_color == Color::kBlack) {
    return point.y >= 0 && point.y <= 4;
  } else if (our_color == Color::kRed) {
    return point.y >= 5 && point.y <= 9;
  } else {
    SpielFatalError(absl::StrCat("Unsupported color: ", our_color));
    return false;
  }
}

std::string ColorToString(Color c) {
  switch (c) {
    case Color::kBlack:
      return "black";
    case Color::kRed:
      return "red";
    case Color::kEmpty:
      return "empty";
    default:
      SpielFatalError(absl::StrCat("Unknown color: ", c));
      return "This will never return.";
  }
}

// Piece implementation
std::string PieceTypeToString(PieceType p) {
  switch (p) {
    case PieceType::kEmpty:
      return " ";
    case PieceType::kPawn:
      return "P";
    case PieceType::kKnight:
      return "N";
    case PieceType::kBishop:
      return "B";
    case PieceType::kRook:
      return "R";
    case PieceType::kAdvisor:
      return "A";
    case PieceType::kKing:
      return "K";
    case PieceType::kCannon:
      return "C";
    default:
      SpielFatalError("Unknown piece.");
      return "This will never return.";
  }
}

std::string Piece::ToString() const {
  std::string base = PieceTypeToString(type);
  return color == Color::kRed ? absl::AsciiStrToUpper(base) : absl::AsciiStrToLower(base);
}

// Board implementation
Board::Board()
    : to_play_(Color::kRed),
      move_number_(1),
      zobrist_hash_(0) {
  board_.fill(kEmptyPiece);
}

void Board::SetPiece(Point p, Piece piece) {
  board_[p.index] = piece;
}

void Board::SetToPlay(Color c) {
  // static const ZobristTableU64<2> kZobristValues(/*seed=*/284628);

  // Remove old color and add new to play.
  // zobrist_hash_ ^= kZobristValues[ToInt(to_play_)];
  // zobrist_hash_ ^= kZobristValues[ToInt(c)];
  to_play_ = c;
}

void Board::SetMovenumber(int move_number) {
  move_number_ = move_number;
}

std::string Board::ToFEN() const {
  std::string fen;

  // 1. encode the board.
  for (int8_t y = kBoardRows-1; y >=0; --y) {
    int num_empty = 0;
    for (int8_t x = 0; x < kBoardCols; ++x) {
      auto piece = at(Point{x, y}.index);
      if (piece == kEmptyPiece) {
        ++num_empty;
      } else {
        if (num_empty > 0) {
          fen += std::to_string(num_empty);
          num_empty = 0;
        }
        fen += piece.ToString();
      }
    }
    if (num_empty > 0) {
      fen += std::to_string(num_empty);
    }

    if (y > 0) {
      fen += "/";
    }
  }

  // 2. color to play.
  fen +=
      " " + (to_play_ == Color::kRed ? std::string("w") : std::string("b"));

  // 3. by castling rights.
  fen += " ";
  fen += "-";

  // 4. en passant square
  fen += " ";
  fen += "-";

  // 5. half-move clock for 50-move rule
  fen += " ";
  fen += "0";

  // 6. full-move clock
  fen += " " + std::to_string(move_number_);

  return fen;
}

std::string Board::DebugString() const {
  std::string s;
  s = absl::StrCat("FEN: ", ToFEN(), "\n");
  // absl::StrAppend(&s, "\n  ---------------------------------\n");
  for (int8_t y = kBoardRows-1; y >= 0; --y) {
    // Row label.
    absl::StrAppend(&s, RowToString(y), " ");

    // Pieces on the row.
    for (int8_t x = 0; x < kBoardCols; ++x) {
      absl::StrAppend(&s, "|", at(Point{x,y}.index).ToString());
    }
    absl::StrAppend(&s, "|\n");
    // absl::StrAppend(&s, "  ---------------------------------\n");
  }

  // Col labels.
  absl::StrAppend(&s, "   ");
  for (int8_t x = 0; x < kBoardCols; ++x) {
    absl::StrAppend(&s, ColToString(x), " ");
  }
  absl::StrAppend(&s, "\n");

  absl::StrAppend(&s, "To play: ", to_play_ == Color::kRed ? "W" : "B", "\n");
  absl::StrAppend(&s, "Move number: ", move_number_, "\n\n");
  absl::StrAppend(&s, "\n");
  return s;
}

template <typename YieldFn>
void Board::GenerateKingDestinations_(
    Point point, Color color, const YieldFn &yield) const {
  static const std::array<Offset, 8> kOffsets = {
      {{1, 0}, {0, 1}, {0, -1}, {-1, 0}}};

  for (const auto &offset : kOffsets) {
    Point dest = point + offset;
    if (InKingsArea(dest) && IsEmptyOrEnemy(dest, color)) {
      yield(dest);
    }
  }
}

template <typename YieldFn>
void Board::GenerateAdvisorDestinations_(
    Point point, Color color, const YieldFn &yield) const {
  static const std::array<Offset, 8> kOffsets = {
      {{-1, 1}, {1, 1}, {-1, -1}, {1, -1}}};

  for (const auto &offset : kOffsets) {
    Point dest = point + offset;
    if (InKingsArea(dest) && IsEmptyOrEnemy(dest, color)) {
      yield(dest);
    }
  }
}

template <typename YieldFn>
void Board::DoGenerateRookDestinations_(
    Point point, Color color, Offset offset_step, const YieldFn &yield) const {
  for (Point dest = point + offset_step; InBoardArea(dest); dest += offset_step) {
    if (IsEmpty(dest)) {
      yield(dest);
    } else if (IsEnemy(dest, color)) {
      yield(dest);
      break;
    } else {
      // We have a friendly piece.
      break;
    }
  }
}

template <typename YieldFn>
void Board::GenerateRookDestinations_(
    Point point, Color color, const YieldFn &yield) const {
  DoGenerateRookDestinations_(point, color, {1, 0}, yield);
  DoGenerateRookDestinations_(point, color, {-1, 0}, yield);
  DoGenerateRookDestinations_(point, color, {0, 1}, yield);
  DoGenerateRookDestinations_(point, color, {0, -1}, yield);
}

template <typename YieldFn>
void Board::GenerateBishopDestinations_(
    Point point, Color color, const YieldFn &yield) const {
  for (int i = 0; i < kBishopOffsets.size(); ++i) {
    const auto &offset = kBishopOffsets[i];
    const auto &blocker_offset = kBishopBlockers[i];
    Point dest = point + offset;
    Point blocker = point + blocker_offset;
    if (InBoardArea(dest) && IsEmptyOrEnemy(dest, color) &&
        IsEmpty(blocker) && !InEnemyArea(dest, color)) {
      yield(dest);
    }
  }
}

template <typename YieldFn>
void Board::GenerateKnightDestinations_(
    Point point, Color color, const YieldFn &yield) const {
  for (int i = 0; i < kKnightOffsets.size(); ++i) {
    const auto &offset = kKnightOffsets[i];
    const auto &blocker_offset = kKnightBlockers[i];
    Point dest = point + offset;
    Point blocker = point + blocker_offset;
    if (InBoardArea(dest) && IsEmptyOrEnemy(dest, color) && IsEmpty(blocker)) {
      yield(dest);
    }
  }
}

template <typename YieldFn>
void Board::DoGenerateCannonDestinations_(
    Point point, Color color, Offset offset_step, const YieldFn &yield) const {
  bool hasCannonPal = false;
  for (Point dest = point + offset_step; InBoardArea(dest); dest += offset_step) {
    if (IsEmpty(dest)) {
      // Move to empty point
      if (!hasCannonPal) {
        yield(dest);
      }
    } else {
      if (hasCannonPal) {
        // Attach enemy over another piece
        if (IsEnemy(dest, color)) {
          yield(dest);
          break;
        } else if (IsFriendly(dest, color)) {
          break;
        }
      } else {
        hasCannonPal = true;
      }
    }
  }
}

template <typename YieldFn>
void Board::GenerateCannonDestinations_(
    Point point, Color color, const YieldFn &yield) const {
  DoGenerateCannonDestinations_(point, color, {1, 0}, yield);
  DoGenerateCannonDestinations_(point, color, {-1, 0}, yield);
  DoGenerateCannonDestinations_(point, color, {0, 1}, yield);
  DoGenerateCannonDestinations_(point, color, {0, -1}, yield);
}

template <typename YieldFn>
void Board::GeneratePawnDestinations_(
    Point point, Color color, const YieldFn &yield) const {
  for (const auto &offset : kPawnOffsets) {
    if (!InEnemyArea(point, color)) {
      if (offset.x_offset != 0)
        continue;
    }

    Point dest = point + offset;
    if (InBoardArea(dest) && IsEmptyOrEnemy(dest, color)) {
      yield(dest);
    }
  }
}

void Board::GenerateLegalMoves(const MoveYieldFn& yield) const {
  bool generating = true;

#define YIELD(move)     \
  if (!yield(move)) {   \
    generating = false; \
  }

  for (int8_t y = 0; y < kBoardRows && generating; ++y) {
    for (int8_t x = 0; x < kBoardCols && generating; ++x) {
      Point point{x, y};
      auto &piece = at(point);
      if (piece.type != PieceType::kEmpty && piece.color == to_play_) {
        switch (piece.type) {
          case PieceType::kKing:
            GenerateKingDestinations_(
                point, to_play_,
                [&yield, &piece, &point, &generating](const Point &to) {
                  YIELD(Move(point, to, piece));
                });
            break;
          case PieceType::kAdvisor:
            GenerateAdvisorDestinations_(
                point, to_play_,
                [&yield, &piece, &point, &generating](const Point &to) {
                  YIELD(Move(point, to, piece));
                });
            break;
          case PieceType::kRook:
            GenerateRookDestinations_(
                point, to_play_,
                [&yield, &piece, &point, &generating](const Point &to) {
                  YIELD(Move(point, to, piece));
                });
            break;
          case PieceType::kBishop:
            GenerateBishopDestinations_(
                point, to_play_,
                [&yield, &piece, &point, &generating](const Point &to) {
                  YIELD(Move(point, to, piece));
                });
            break;
          case PieceType::kKnight:
            GenerateKnightDestinations_(
                point, to_play_,
                [&yield, &piece, &point, &generating](const Point &to) {
                  YIELD(Move(point, to, piece));
                });
            break;
          case PieceType::kCannon:
            GenerateCannonDestinations_(
                point, to_play_,
                [&yield, &piece, &point, &generating](const Point &to) {
                  YIELD(Move(point, to, piece));
                });
            break;
          case PieceType::kPawn:
            GeneratePawnDestinations_(
                point, to_play_,
                [&yield, &piece, &point, &generating](const Point &to) {
                  YIELD(Move(point, to, piece));
                });
            break;
          default:
            std::cerr << "Unknown piece type: " << static_cast<int>(piece.type)
                      << std::endl;
        }
      }
    }
  }

#undef YIELD
}

// Helper functions

absl::optional<PieceType> PieceTypeFromChar(char c) {
  switch (toupper(c)) {
    case 'P':
      return PieceType::kPawn;
    case 'N':
      return PieceType::kKnight;
    case 'B':
      return PieceType::kBishop;
    case 'R':
      return PieceType::kRook;
    case 'A':
      return PieceType::kAdvisor;
    case 'K':
      return PieceType::kKing;
    case 'C':
      return PieceType::kCannon;
    default:
      std::cerr << "Invalid piece type: " << c << std::endl;
      return std::nullopt;
  }
}

inline absl::optional<Board> CreateBoard(const std::string &fen) {
  Board board;
  std::vector<std::string> fen_parts = absl::StrSplit(fen, ' ');
  if (fen_parts.size() != 6 && fen_parts.size() != 4 && fen_parts.size() != 2) {
    std::cerr << "Invalid FEN: " << fen << std::endl;
    return std::nullopt;
  }

  std::string &piece_configuration = fen_parts[0];
  std::string &side_to_move = fen_parts[1];

  if (fen_parts.size() >= 4) {
    std::string &castling_rights = fen_parts[2];
    std::string &ep_square = fen_parts[3];
  }

  // These are defaults if the FEN string doesn't have these fields.
  std::string fifty_clock = "0";
  std::string move_number = "1";

  if (fen_parts.size() == 6) {
    fifty_clock = fen_parts[4];
    move_number = fen_parts[5];
  }

  std::vector<std::string> piece_config_by_row =
      absl::StrSplit(piece_configuration, '/');

  for (int8_t y = kBoardRows-1; y >=0; --y) {
    std::string &row = piece_config_by_row[kBoardRows-y-1];
    int8_t x = 0;
    for (char c : row) {
      if (x >= kBoardCols) {
        std::cerr << "Too many things on FEN row: " << row << std::endl;
        return std::nullopt;
      }

      if (c >= '1' && c <= '9') {
        x += c - '0';
      } else {
        auto piece_type = PieceTypeFromChar(c);
        if (!piece_type) {
          std::cerr << "Invalid piece type in FEN: " << c << std::endl;
          return std::nullopt;
        }

        Color color = isupper(c) ? Color::kRed : Color::kBlack;
        board.SetPiece(Point{x, y}, Piece{color, *piece_type});
        ++x;
      }
    }
  }

  if (side_to_move == "b") {
    board.SetToPlay(Color::kBlack);
  } else if (side_to_move == "w") {
    board.SetToPlay(Color::kRed);
  } else {
    board.SetToPlay(Color::kRed);
  }

  board.SetMovenumber(std::stoi(move_number));

  return board;
}

Board MakeDefaultBoard() {
  // FEN spec: http://www.xqbase.com/protocol/cchess_fen.htm
  auto maybe_board = CreateBoard(
      "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1");
  SPIEL_CHECK_TRUE(maybe_board);
  return *maybe_board;
}

}  // namespace chinese_chess
}  // namespace open_spiel
