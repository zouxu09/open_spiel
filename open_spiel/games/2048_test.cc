// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "open_spiel/games/2048.h"

#include "open_spiel/spiel.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace two_zero_four_eight {
namespace {

namespace testing = open_spiel::testing;

void BasicSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void RandomSerializationTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  for (int i = 0; i < 20; ++i) {
    state->ApplyAction(state->LegalActions()[0]);
  }
  std::unique_ptr<State> state2 = game->DeserializeState(state->Serialize());
  SPIEL_CHECK_EQ(state->ToString(), state2->ToString());
}

void Basic2048Tests() {
  testing::LoadGameTest("2048");
  testing::ChanceOutcomesTest(*LoadGame("2048"));
  testing::RandomSimTest(*LoadGame("2048"), 100);
}

// Board:
//    0    0    0    0
//    2    0    0    0
//    2    0    0    0
//    2    0    0    0
// 4 should be formed in the bottom left corner and not on the cell above it
void MultipleMergeTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwoZeroFourEightState* cstate = 
      static_cast<TwoZeroFourEightState*>(state.get());
  cstate->SetCustomBoard({0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0});
  cstate->ApplyAction(cstate->LegalActions()[2]);
  SPIEL_CHECK_EQ(cstate->BoardAt(3, 0).value, 4);  
}

// Board:
//    4    8    2    4
//    2    4    8   16
//   16  128   64  128
//    2    8    2    8
// This should be a losing terminal state
void TerminalStateTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwoZeroFourEightState* cstate = 
      static_cast<TwoZeroFourEightState*>(state.get());
  cstate->SetCustomBoard(
      {4, 8, 2, 4, 2, 4, 8, 16, 16, 128, 64, 128, 2, 8, 2, 8});
  SPIEL_CHECK_EQ(cstate->IsTerminal(), true);
  SPIEL_CHECK_EQ(cstate->Returns()[0], -1.0); 
}

// Board:
//    4    8    2    4
//    2    4    8   16
// 1024  128   64  128
// 1024    8    2    8
// Taking down action should win from this state
void GameWonTest() {
  std::shared_ptr<const Game> game = LoadGame("2048");
  std::unique_ptr<State> state = game->NewInitialState();
  TwoZeroFourEightState* cstate = 
      static_cast<TwoZeroFourEightState*>(state.get());
  cstate->SetCustomBoard(
      {4, 8, 2, 4, 2, 4, 8, 16, 1024, 128, 64, 128, 1024, 8, 2, 8});
  cstate->ApplyAction(cstate->LegalActions()[2]);
  SPIEL_CHECK_EQ(cstate->IsTerminal(), true);
  SPIEL_CHECK_EQ(cstate->Returns()[0], 1.0);
}

}  // namespace
}  // namespace two_zero_four_eigth
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::two_zero_four_eight::BasicSerializationTest();
  open_spiel::two_zero_four_eight::RandomSerializationTest();
  open_spiel::two_zero_four_eight::Basic2048Tests();
  open_spiel::two_zero_four_eight::MultipleMergeTest();
  open_spiel::two_zero_four_eight::TerminalStateTest();
  open_spiel::two_zero_four_eight::GameWonTest();
}
