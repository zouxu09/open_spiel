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
#include <bitset>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h"
#include "open_spiel/games/chinese_chess/board.h"

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

// Adds a plane to the information state vector corresponding to the presence
// and absence of the given piece type and colour at each square.
void AddPieceTypePlane(
  Color color, PieceType piece_type, const Board& board,
  absl::Span<float>::iterator& value_it) {
  for (int8_t y = 0; y < kBoardRows; ++y) {
    for (int8_t x = 0; x < kBoardCols; ++x) {
      Piece piece_on_board = board.at(Point{x, y});
      *value_it++ =
          (piece_on_board.color == color && piece_on_board.type == piece_type
               ? 1.0
               : 0.0);
    }
  }
}

// Adds a uniform scalar plane scaled with min and max.
template <typename T>
void AddScalarPlane(
  T val, T min, T max, absl::Span<float>::iterator& value_it) {
  double normalized_val = static_cast<double>(val - min) / (max - min);
  for (int i = 0; i < kBoardRows * kBoardCols; ++i)
    *value_it++ = normalized_val;
}

} // namespace

Action MoveToAction(const Move& move) {
  std::bitset<7> from(move.from.index);
  std::bitset<7> to(move.to.index);
  std::bitset<3> piece((uint8_t)move.piece.type);
  std::bitset<2> color((uint8_t)move.piece.color);

  std::bitset<19> action;
  int k = 0;
  for (int i = 0; i < from.size(); ++i, ++k) {
    action[k] = from[i];
  }
  for (int i = 0; i < to.size(); ++i, ++k) {
    action[k] = to[i];
  }
  for (int i = 0; i < piece.size(); ++i, ++k) {
    action[k] = piece[i];
  }
  for (int i = 0; i < color.size(); ++i, ++k) {
    action[k] = color[i];
  }

  return action.to_ulong();
}

Move ActionToMove(const Action& action) {
  SPIEL_CHECK_GE(action, 0);
  SPIEL_CHECK_LT(action, kNumDistinctActions);

  std::bitset<19> actionBitset(action);

  std::bitset<7> from;
  std::bitset<7> to;
  std::bitset<3> piece;
  std::bitset<2> color;

  int k = 0;
  for (int i = 0; i < from.size(); ++i, ++k) {
    from[i] = actionBitset[k];
  }
  for (int i = 0; i < to.size(); ++i, ++k) {
    to[i] = actionBitset[k];
  }
  for (int i = 0; i < piece.size(); ++i, ++k) {
    piece[i] = actionBitset[k];
  }
  for (int i = 0; i < color.size(); ++i, ++k) {
    color[i] = actionBitset[k];
  }

  Move move(from.to_ulong(), to.to_ulong(),
    Piece{Color{color.to_ulong()}, PieceType{piece.to_ulong()}});
  return move;
}

// ChineseChessState
void ChineseChessState::DoApplyAction(Action action) {
  Move move = ActionToMove(action);
  moves_history_.push_back(move);
  CurrentBoard().ApplyMove(move);
  // ++repetitions_[current_board_.HashValue()];
  cached_legal_actions_.reset();
}

std::vector<Action> ChineseChessState::LegalActions() const {
  MaybeGenerateLegalActions();
  if (IsTerminal()) return {};
  return *cached_legal_actions_;
}

std::string ChineseChessState::ActionToString(
  Player player, Action action_id) const {
  Move move = ActionToMove(action_id);
  return move.ToString();
}

ChineseChessState::ChineseChessState(std::shared_ptr<const Game> game)
  : State(game),
    start_board_(MakeDefaultBoard()),
    current_board_(start_board_) {
}

std::string ChineseChessState::ToString() const {
  return CurrentBoard().ToFEN();
}

std::vector<double> ChineseChessState::Returns() const {
  auto maybe_final_returns = MaybeFinalReturns();
  if (maybe_final_returns) {
    return *maybe_final_returns;
  } else {
    return {0.0, 0.0};
  }
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

  auto value_it = values.begin();

  // Piece cconfiguration.
  for (const auto& piece_type : kPieceTypes) {
    AddPieceTypePlane(Color::kRed, piece_type, CurrentBoard(), value_it);
    AddPieceTypePlane(Color::kBlack, piece_type, CurrentBoard(), value_it);
  }

  // Empty points
  AddPieceTypePlane(Color::kEmpty, PieceType::kEmpty, CurrentBoard(), value_it);

  // Side to play.
  AddScalarPlane(ColorToPlayer(CurrentBoard().ToPlay()), 0, 1, value_it);

  SPIEL_CHECK_EQ(value_it, values.end());
}

void ChineseChessState::UndoAction(Player player, Action move) {
}

std::unique_ptr<State> ChineseChessState::Clone() const {
  return std::unique_ptr<State>(new ChineseChessState(*this));
}

void ChineseChessState::MaybeGenerateLegalActions() const {
  if (!cached_legal_actions_) {
    cached_legal_actions_ = std::vector<Action>();
    CurrentBoard().GenerateLegalMoves([this](const Move& move) -> bool {
      cached_legal_actions_->push_back(MoveToAction(move));
      return true;
    });
    absl::c_sort(*cached_legal_actions_);
  }
}

absl::optional<std::vector<double>> ChineseChessState::MaybeFinalReturns() const {
  if (CurrentBoard().GetMovenumber() > 100) {
    return std::vector<double>{DrawUtility(), DrawUtility()};
  }

  if (CurrentBoard().CheckMate()) {
    std::vector<double> returns(NumPlayers());
    auto next_to_play = ColorToPlayer(CurrentBoard().ToPlay());
    returns[next_to_play] = WinUtility();
    returns[OtherPlayer(next_to_play)] = LossUtility();
    return returns;
  }

  // Compute and cache the legal actions.
  // MaybeGenerateLegalActions();
  // SPIEL_CHECK_TRUE(cached_legal_actions_);
  // bool have_legal_moves = !cached_legal_actions_->empty();

  // if (!have_legal_moves) {
  //   std::vector<double> returns(NumPlayers());
  //   auto next_to_play = ColorToPlayer(CurrentBoard().ToPlay());
  //   returns[next_to_play] = LossUtility();
  //   returns[OtherPlayer(next_to_play)] = WinUtility();
  //   return returns;
  // }

  return std::nullopt;
}

// ChineseChessGame
ChineseChessGame::ChineseChessGame(const GameParameters& params)
    : Game(kGameType, params) {}

}  // namespace chinese_chess
}  // namespace open_spiel
