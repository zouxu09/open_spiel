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

#ifndef OPEN_SPIEL_GAMES_CHINESE_CHESS_BOARD_H_
#define OPEN_SPIEL_GAMES_CHINESE_CHESS_BOARD_H_

#include <array>
#include <cstdint>
#include <ostream>
#include <vector>

namespace open_spiel {
namespace chinese_chess {

enum class Color : uint8_t { kRed = 0, kBlack = 1, kEmpty = 2 };

std::string ColorToString(Color c);

std::ostream &operator<<(std::ostream &os, Color c);

Color OppColor(Color c);

using Point = uint16_t;

inline constexpr int kBoardRows = 10;
inline constexpr int kBoardCols = 9;
inline constexpr int kBoardPoints = 90;

// Returns a reference to a vector that contains all points that are on a board
const std::vector<Point> &BoardPoints();

inline Point ToPoint(int x, int y) {
  return y * kBoardCols + x;
}

enum class PieceType : int8_t {
  kEmpty = 0,
  kKing = 1,
  kAdvisor = 2,
  kRook = 3,
  kBishop = 4,
  kKnight = 5,
  kCannon = 6,
  kPawn = 7
};

struct Piece {
  bool operator==(const Piece& other) const {
    return type == other.type && color == other.color;
  }

  bool operator!=(const Piece& other) const { return !(*this == other); }

  // std::string ToUnicode() const;
  std::string ToString() const;

  Color color;
  PieceType type;
};

static inline constexpr Piece kEmptyPiece = Piece{Color::kEmpty, PieceType::kEmpty};

// Simple Chinese Chess board that is optimized for speed.
class Board {
 public:
  explicit Board();

  void Clear();

  inline Color PointColor(Point p) const { return board_[p].color; }

  inline bool IsEmpty(Point p) const {
    return PointColor(p) == Color::kEmpty;
  }

  bool IsLegalMove(Point p, Color c) const;

  bool PlayMove(Point p, Color c);

  inline uint64_t HashValue() const { return zobrist_hash_; }

  std::string ToString();

  const Piece& at(Point p) const { return board_[p]; }

  void SetPiece(Point p, Piece piece);

  Color ToPlay() const { return to_play_; }
  void SetToPlay(Color c);
  void SetMovenumber(int move_number);

  std::string ToFEN() const;

 private:
  std::array<Piece, kBoardPoints> board_;
  Color to_play_;
  // This starts at 1, and increments after each black move (a "full move" in
  // chess is a "half move" by white followed by a "half move" by black).
  int32_t move_number_;
  uint64_t zobrist_hash_;
};

std::ostream &operator<<(std::ostream &os, const Board &board);

Board MakeDefaultBoard();

}  // namespace chinese_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHINESE_CHESS_BOARD_H_
