# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Tests for Python Block Dominoes."""

from absl.testing import absltest
from open_spiel.python.algorithms.get_all_states import get_all_states
import pyspiel
import block_dominoes

class DominoesBlockTest(absltest.TestCase):

  def test_game_from_cc(self):
    """Runs our standard game tests, checking API consistency."""
    game = pyspiel.load_game("python_block_dominoes")
    pyspiel.random_sim_test(game, num_sims=100, serialize=False, verbose=True)

  def test_single_deterministic_game_1(self):
    """Runs a single game where tiles and actions chose deterministically."""
    game = pyspiel.load_game("python_block_dominoes")
    state = game.new_initial_state()
    hand0 = [(6., 6.),(0., 2.),(4., 4.),(3., 3.),(2., 2.),(1., 1.),(0., 0.)]
    hand1  = [(5., 6.),(4., 5.),(3., 4.),(2., 3.),(1., 2.),(0., 1.),(4., 6.)]
    self.deal_hands(state, [hand0, hand1])

    self.apply_action(state, block_dominoes.Action(0, (6., 6.), None))
    self.apply_action(state, block_dominoes.Action(1, (5., 6.), 6.))
    # player 0 don't hold any tile with 6 or 5, player 1 turn again
    self.apply_action(state, block_dominoes.Action(1, (4., 5.), 5.))
    self.apply_action(state, block_dominoes.Action(0, (4., 4.), 4.))
    self.apply_action(state, block_dominoes.Action(1, (3., 4.), 4.))
    self.apply_action(state, block_dominoes.Action(0, (3., 3.), 3.))
    self.apply_action(state, block_dominoes.Action(1, (2., 3.), 3.))
    self.apply_action(state, block_dominoes.Action(0, (2., 2.), 2.))
    self.apply_action(state, block_dominoes.Action(1, (1., 2.), 2.))
    self.apply_action(state, block_dominoes.Action(0, (1., 1.), 1.))
    self.apply_action(state, block_dominoes.Action(1, (0., 1.), 1.))
    self.apply_action(state, block_dominoes.Action(0, (0., 0.), 0.))
    self.apply_action(state, block_dominoes.Action(1, (4., 6.), 6.))

    # player 1 played all is tile and player 0 hold the tile (0, 2)
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns()[0], -2)
    self.assertEqual(state.returns()[1], 2)

  def test_single_deterministic_game_2(self):
    """Runs a single game where tiles and actions chose deterministically."""
    game = pyspiel.load_game("python_block_dominoes")
    state = game.new_initial_state()
    hand0 = [(6., 6.),(0., 5.),(1., 5.),(2., 5.),(3., 5.),(4., 5.),(5., 5.)]
    hand1  = [(0., 4.),(1., 4.),(2., 4.),(3., 4.),(4., 4.),(0., 3.),(1., 3.)]
    self.deal_hands(state, [hand0, hand1])

    self.apply_action(state, block_dominoes.Action(0, (6., 6.), None))
    # Both players don't hold tile with 6, therefore both blocked and the game end
    self.assertTrue(state.is_terminal())
    self.assertEqual(state.returns()[0], -45)
    self.assertEqual(state.returns()[1], 45)

  @staticmethod
  def apply_action(state, action):
    actions_str = block_dominoes._ACTIONS_STR
    state.apply_action(actions_str.index(str(action)))

  @staticmethod
  def deal_hands(state, hands):
    deck = block_dominoes._DECK
    for hand in hands:
      for t in hand:
        state.apply_action(deck.index(t))


if __name__ == "__main__":
    absltest.main()
