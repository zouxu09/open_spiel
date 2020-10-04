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
#include "open_spiel/games/chinese_chess/board.h"

// Chinese Chess (Xiangqi)
// https://en.wikipedia.org/wiki/Xiangqi
//

namespace open_spiel {
namespace chinese_chess {

// Constants.
inline constexpr int kNumPlayers = 2;
inline constexpr int kNumDistinctActions = 524288;
inline constexpr int kMaxGameLength = 512;

inline constexpr double LossUtility() { return -1; }
inline constexpr double DrawUtility() { return 0; }
inline constexpr double WinUtility() { return 1; }

class Move;

Action MoveToAction(const Move& move);
Move ActionToMove(const Action& action);

inline int ColorToPlayer(Color c) {
  if (c == Color::kRed) {
    return 0;
  } else if (c == Color::kBlack) {
    return 1;
  } else {
    SpielFatalError("Unknown color");
  }
}

inline int OtherPlayer(Player player) { return player == Player{0} ? 1 : 0; }

class ChineseChessState : public State {
 public:
  ChineseChessState(std::shared_ptr<const Game> game);

  ChineseChessState(const ChineseChessState&) = default;
  ChineseChessState& operator=(const ChineseChessState&) = default;

  Player CurrentPlayer() const override {
    return IsTerminal() ? kTerminalPlayerId : ColorToPlayer(CurrentBoard().ToPlay());
  }

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

  // Current board.
  Board& CurrentBoard() { return current_board_; }
  const Board& CurrentBoard() const { return current_board_; }

  // Starting board.
  Board& StartBoard() { return start_board_; }
  const Board& StartBoard() const { return start_board_; }

  std::vector<Move>& MovesHistory() { return moves_history_; }
  const std::vector<Move>& MovesHistory() const { return moves_history_; }

  absl::optional<std::vector<double>> MaybeFinalReturns() const;

 protected:
  void DoApplyAction(Action move) override;

 private:
  void MaybeGenerateLegalActions() const;

 private:
  // We have to store every move made to check for repetitions and to implement
  // undo. We store the current board position as an optimization.
  std::vector<Move> moves_history_;
  // We store the start board for history to support games not starting
  // from the start position.
  Board start_board_;
  // We store the current board position as an optimization.
  Board current_board_;

  mutable absl::optional<std::vector<Action>> cached_legal_actions_;
};

// Game object.
class ChineseChessGame : public Game {
 public:
  explicit ChineseChessGame(const GameParameters& params);
  int NumDistinctActions() const override { return kNumDistinctActions; };
  std::unique_ptr<State> NewInitialState() const override {
    return std::unique_ptr<State>(new ChineseChessState(shared_from_this()));
  }
  int NumPlayers() const override { return kNumPlayers; }
  double MinUtility() const override { return -1; }
  double UtilitySum() const override { return 0; }
  double MaxUtility() const override { return 1; }
  std::vector<int> ObservationTensorShape() const override { return {}; };
  int MaxGameLength() const override { return kMaxGameLength; };
};

}  // namespace chinese_chess
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_CHINESE_CHESS_H_
