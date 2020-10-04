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

#include "open_spiel/games/chinese_chess.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"

namespace open_spiel {
namespace chinese_chess {
namespace {

// Facts about the game.
const GameType kGameType{
    /*short_name=*/"chinese_chess",
    /*long_name=*/"Chinese Chess",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/2,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/{}  // no parameters
};

std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new ChineseChessGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

}  // namespace

// ChineseChessState
void ChineseChessState::DoApplyAction(Action move) {
}

std::vector<Action> ChineseChessState::LegalActions() const {
}

Player ChineseChessState::CurrentPlayer() const {
  return kTerminalPlayerId;
}

std::string ChineseChessState::ActionToString(
  Player player, Action action_id) const {
  return "";
}

ChineseChessState::ChineseChessState(std::shared_ptr<const Game> game) : State(game) {
}

std::string ChineseChessState::ToString() const {
  std::string str;
  return str;
}

bool ChineseChessState::IsTerminal() const {
  return true;
}

std::vector<double> ChineseChessState::Returns() const {
  return {0.0, 0.0};
}

std::string ChineseChessState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return HistoryString();
}

std::string ChineseChessState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
  return ToString();
}

void ChineseChessState::ObservationTensor(
  Player player, absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, num_players_);
}

void ChineseChessState::UndoAction(Player player, Action move) {
}

std::unique_ptr<State> ChineseChessState::Clone() const {
  return std::unique_ptr<State>(new ChineseChessState(*this));
}

// ChineseChessGame
ChineseChessGame::ChineseChessGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace chinese_chess
}  // namespace open_spiel
