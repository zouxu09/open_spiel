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

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"
#include "open_spiel/games/chinese_chess.h"
#include "open_spiel/games/chinese_chess/board.h"

namespace open_spiel {
namespace chinese_chess {
namespace {

namespace testing = open_spiel::testing;

int CountNumLegalMoves(const Board& board) {
  int num_legal_moves = 0;
  board.GenerateLegalMoves([&num_legal_moves](const Move& move) -> bool {
    ++num_legal_moves;
    std::cout << move << " ";
    return true;
  });
  return num_legal_moves;
}

void BasicChineseChessTests() {
  testing::LoadGameTest("chinese_chess");
  testing::NoChanceOutcomesTest(*LoadGame("chinese_chess"));
  testing::RandomSimTest(*LoadGame("chinese_chess"), 10);
}

void FENGenerateAndParseTests() {
  Board root_state = MakeDefaultBoard();
  SPIEL_CHECK_EQ(root_state.ToFEN(), "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1");
  std::cout << root_state.DebugString();
}

void PointTests() {
  Point p1{3, 4};
  SPIEL_CHECK_EQ(p1.x, 3);
  SPIEL_CHECK_EQ(p1.y, 4);
  SPIEL_CHECK_EQ(p1.index, 39);

  Point p2{39};
  SPIEL_CHECK_EQ(p1, p2);

  Offset offset{1, 1};
  p2 += offset;
  SPIEL_CHECK_EQ(p2.x, 4);
  SPIEL_CHECK_EQ(p2.y, 5);
}

void BoardTests() {
  Board root_state = MakeDefaultBoard();
  std::cout << "Piece is: " << (u_int)root_state.at(Point{0, 3}).type << std::endl;
  std::cout << "Check if is empty: " << root_state.IsEmpty(Point{0, 3}) << std::endl;
}

void MoveGenerationTests() {
  Board root_state = MakeDefaultBoard();
  std::cout  << "Legal moves: " << CountNumLegalMoves(root_state) << std::endl;
  // SPIEL_CHECK_EQ(CountNumLegalMoves(root_state), 20);
}

void ActionsTests() {
  Move move(Point{0, 0}, Point{0, 1}, Piece{Color::kRed, PieceType::kRook});
  Action action = MoveToAction(move);
  std::cout << action << std::endl;

  Move m = ActionToMove(action);
  std::cout << m << std::endl;
}

void StateActionTests() {
  std::shared_ptr<const Game> game = LoadGame("chinese_chess");
  auto state = game->NewInitialState();
  state->ObservationTensor();

  while (!state->IsTerminal()) {
    int random_num = rand();
    auto legal_actions = state->LegalActions();
    random_num = random_num % legal_actions.size();
    auto action = legal_actions[random_num];
    state->ApplyAction(action);
    std::cout << state->ToString() << std::endl;
  }
}

}  // namespace
}  // namespace chinese_chess
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::chinese_chess::BasicChineseChessTests();
  // open_spiel::chinese_chess::FENGenerateAndParseTests();
  // open_spiel::chinese_chess::PointTests();
  // open_spiel::chinese_chess::BoardTests();
  // open_spiel::chinese_chess::ActionsTests();
  // open_spiel::chinese_chess::MoveGenerationTests();
  // open_spiel::chinese_chess::StateActionTests();
}
