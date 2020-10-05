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

inline Color OppColor(Color color) {
  return color == Color::kRed ? Color::kBlack : Color::kRed;
}

inline constexpr int kBoardRows = 10;
inline constexpr int kBoardCols = 9;
inline constexpr int kBoardPoints = 90;

struct Offset {
  int8_t x_offset;
  int8_t y_offset;

  bool operator==(const Offset& other) const {
    return x_offset == other.x_offset && y_offset == other.y_offset;
  }
};

// Offsets for all possible knight moves.
inline constexpr std::array<Offset, 8> kKnightOffsets = {
    {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {2, -1}, {2, 1}, {1, -2}, {1, 2}}};
// Offsets for all posiable knight blockers according to moves
inline constexpr std::array<Offset, 8> kKnightBlockers = {
    {{-1, 0}, {-1, 0}, {0, -1}, {0, 1}, {1, 0}, {1, 0}, {0, -1}, {0, 1}}};

// Offsets for all possible bishop moves.
inline constexpr std::array<Offset, 4> kBishopOffsets = {
    {{-2, -2}, {-2, 2}, {2, 2}, {2, -2}}};
// Offsets for all posiable bishop blockers according to moves
inline constexpr std::array<Offset, 4> kBishopBlockers = {
    {{-1, -1}, {-1, 1}, {1, 1}, {1, -1}}};

// Offsets for all possible pawn moves.
inline constexpr std::array<Offset, 3> kPawnOffsets = {
    {{-1, 0}, {1, 0}, {0, 1}}};

struct Point {
  constexpr Point(int x, int y) : x(x), y(y), index(y*kBoardCols + x) {}
  constexpr Point(int index) : x(index%kBoardCols), y(index/kBoardCols), index(index) {}

  Point& operator+=(const Offset& offset) {
    x += offset.x_offset;
    y += offset.y_offset;
    index = y * kBoardCols + x;
    return *this;
  }

  bool operator==(const Point& other) const {
    return x == other.x && y == other.y;
  }

  bool operator!=(const Point& other) const { return !(*this == other); }

  std::string ToString() const {
    std::string s;
    s.push_back('a' + x);
    s.push_back('0' + y);
    return s;
  }

  // fields
  uint8_t x;
  uint8_t y;
  uint8_t index;
};

inline Point operator+(const Point& point, const Offset& offset) {
  int8_t x = point.x + offset.x_offset;
  int8_t y = point.y + offset.y_offset;
  return Point{x, y};
}

inline std::string RowToString(int8_t row) {
  return std::string(1, '0' + row);
}

inline std::string ColToString(int8_t col) {
  return std::string(1, 'a' + col);
}

inline constexpr Point InvalidPoint() { return Point{-1, -1}; }

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

static inline constexpr std::array<PieceType, 7> kPieceTypes = {
  {PieceType::kKing, PieceType::kAdvisor, PieceType::kRook,
   PieceType::kBishop,PieceType::kKnight, PieceType::kCannon, PieceType::kPawn}};

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

struct Move {
  Point from;
  Point to;
  Piece piece;

  Move(const Point& from, const Point& to, const Piece& piece)
    : from(from), to(to), piece(piece) {}

  std::string ToString() const {
    return from.ToString() + to.ToString();
  };

  bool operator==(const Move& other) const {
    return from == other.from && to == other.to && piece == other.piece;
  }
};

// Simple Chinese Chess board that is optimized for speed.
class Board {
 public:
  explicit Board();

  void Clear();

  inline Color PointColor(Point p) const { return board_[p.index].color; }

  bool IsLegalMove(Point p, Color c) const;

  bool PlayMove(Point p, Color c);

  inline uint64_t HashValue() const { return zobrist_hash_; }

  const Piece& at(Point p) const { return board_[p.index]; }

  void SetPiece(Point p, Piece piece);

  Color ToPlay() const { return to_play_; }
  void SetToPlay(Color c);
  void SetMovenumber(int move_number);
  int GetMovenumber() const { return move_number_; }

  std::string ToFEN() const;
  std::string DebugString() const;

  static bool InBoardArea(const Point& point) {
    return point.x >= 0 && point.x < kBoardCols && point.y >= 0 && point.y < kBoardRows;
  }

  static bool InKingsArea(const Point& point) {
    return (point.x >= 3 && point.x <= 5) &&
      ((point.y >= 0 && point.y <= 2) || (point.y >= 7 && point.y <= 9));
  }

  bool IsEmpty(const Point& point) const {
    return board_[point.index].color == Color::kEmpty;
  }

  bool IsEnemy(const Point& point, Color our_color) const {
    const Piece& piece = board_[point.index];
    return piece.type != PieceType::kEmpty && piece.color != our_color;
  }

  bool IsFriendly(const Point& point, Color our_color) const {
    const Piece& piece = board_[point.index];
    return piece.color == our_color;
  }

  bool IsEmptyOrEnemy(const Point& point, Color our_color) const {
    const Piece& piece = board_[point.index];
    return piece.color != our_color;
  }

  bool IsKingCheck() const;
  bool CheckMate() const;

  using MoveYieldFn = std::function<bool(const Move&)>;
  void GenerateLegalMoves(const MoveYieldFn& yield) const;

  void ApplyMove(const Move &move);

 private:
  template <typename YieldFn>
  void GenerateKingDestinations_(
    Point point, Color color, const YieldFn &yield) const;

  template <typename YieldFn>
  void GenerateAdvisorDestinations_(
    Point point, Color color, const YieldFn &yield) const;

  template <typename YieldFn>
  void GenerateRookDestinations_(
    Point point, Color color, const YieldFn &yield) const;

  template <typename YieldFn>
  void GenerateBishopDestinations_(
    Point point, Color color, const YieldFn &yield) const;

  template <typename YieldFn>
  void GenerateKnightDestinations_(
    Point point, Color color, const YieldFn &yield) const;

  template <typename YieldFn>
  void GenerateCannonDestinations_(
    Point point, Color color, const YieldFn &yield) const;

  template <typename YieldFn>
  void GeneratePawnDestinations_(
    Point point, Color color, const YieldFn &yield) const;

  template <typename YieldFn>
  void DoGenerateRookDestinations_(
    Point point, Color color, Offset offset_step, const YieldFn& yield) const;

  template <typename YieldFn>
  void DoGenerateCannonDestinations_(
    Point point, Color color, Offset offset_step, const YieldFn& yield) const;

  Point find(const Piece &piece) const;

 private:
  std::array<Piece, kBoardPoints> board_;
  Color to_play_;
  // This starts at 1, and increments after each black move (a "full move" in
  // chess is a "half move" by white followed by a "half move" by black).
  int32_t move_number_;
  uint64_t zobrist_hash_;
};

inline std::ostream &operator<<(std::ostream &os, Color c) {
  return os << ColorToString(c);
}

inline std::ostream &operator<<(std::ostream &os, const Board &board) {
  return os << board.DebugString();
}

inline std::ostream& operator<<(std::ostream& os, const Point& point) {
  return os << point.ToString();
}

inline std::ostream& operator<<(std::ostream& os, const Piece& piece) {
  return os << piece.ToString();
}

inline std::ostream& operator<<(std::ostream& os, const Move& move) {
  return os << move.ToString();
}

Board MakeDefaultBoard();

}  // namespace chinese_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHINESE_CHESS_BOARD_H_
