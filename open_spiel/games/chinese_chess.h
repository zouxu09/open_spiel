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

#ifndef OPEN_SPIEL_GAMES_CHINESE_CHESS_H_
#define OPEN_SPIEL_GAMES_CHINESE_CHESS_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/spiel.h"

// Chinese Chess (Xiangqi)
// https://en.wikipedia.org/wiki/Xiangqi
//

namespace open_spiel {
namespace chinese_chess {

// Constants.
inline constexpr int kNumPlayers = 2;

class ChineseChessState : public State {
 public:
  ChineseChessState(std::shared_ptr<const Game> game);

  ChineseChessState(const ChineseChessState&) = default;
  ChineseChessState& operator=(const ChineseChessState&) = default;

  Player CurrentPlayer() const override;
  std::string ActionToString(Player player, Action action_id) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player,
                         absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action move) override;
  std::vector<Action> LegalActions() const override;

 protected:
  void DoApplyAction(Action move) override;
};

// Game object.
class ChineseChessGame : public Game {
 public:
  explicit ChineseChessGame(const GameParameters& params);
  int NumDistinctActions() const override;
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ChineseChessState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override;
  int MaxGameLength() const override;
};

}  // namespace chinese_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHINESE_CHESS_H_
