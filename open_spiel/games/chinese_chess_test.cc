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
#include "open_spiel/games/chinese_chess/board.h"

namespace open_spiel {
namespace chinese_chess {
namespace {

namespace testing = open_spiel::testing;

void BasicChineseChessTests() {
  testing::LoadGameTest("chinese_chess");
  testing::NoChanceOutcomesTest(*LoadGame("chinese_chess"));
  testing::RandomSimTest(*LoadGame("chinese_chess"), 100);
}

void FENGenerateAndParseTests() {
  Board root_state = MakeDefaultBoard();
  SPIEL_CHECK_EQ(root_state.ToFEN(), "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1");

  std::cout << root_state.DebugString();
}

}  // namespace
}  // namespace chinese_chess
}  // namespace open_spiel

int main(int argc, char** argv) {
  // open_spiel::chinese_chess::BasicChineseChessTests();
  open_spiel::chinese_chess::FENGenerateAndParseTests();
}
