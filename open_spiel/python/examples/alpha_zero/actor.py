# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Starting point for playing with the AlphaZero algorithm."""

from absl import app
from absl import flags
import itertools
import random
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as evaluator_lib
from open_spiel.python.utils import file_logger

from open_spiel.python import games  # pylint: disable=unused-import
import pyspiel
import json
import os
import sys
import numpy as np
import tempfile
import datetime
import reverb
import tensorflow as tf
import tree

flags.DEFINE_string("game", "python_xiang", "Name of the game.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("max_simulations", 300, "How many simulations to run.")
flags.DEFINE_integer("train_batch_size", 2 ** 10, "Batch size for learning.")
flags.DEFINE_integer("replay_buffer_size", 2 ** 16,
                     "How many states to store in the replay buffer.")
flags.DEFINE_integer("replay_buffer_reuse", 3,
                     "How many times to learn from each state.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_float("policy_epsilon", 0.25, "What noise epsilon to use.")
flags.DEFINE_float("policy_alpha", 1, "What dirichlet noise alpha to use.")
flags.DEFINE_float("temperature", 1,
                   "Temperature for final move selection.")
flags.DEFINE_integer("temperature_drop", 10,  # Less than AZ due to short games.
                     "Drop the temperature to 0 after this many moves.")
flags.DEFINE_enum("nn_model", "resnet", model_lib.Model.valid_model_types,
                  "What type of model should be used?.")
flags.DEFINE_integer("nn_width", 2 ** 7, "How wide should the network be.")
flags.DEFINE_integer("nn_depth", 10, "How deep should the network be.")
flags.DEFINE_string("path", "./xiang", "Where to save checkpoints.")
flags.DEFINE_integer("checkpoint_freq", 100, "Save a checkpoint every N steps.")
flags.DEFINE_integer("actors", 2, "How many actors to run.")
flags.DEFINE_integer("evaluators", 1, "How many evaluators to run.")
flags.DEFINE_integer("evaluation_window", 100,
                     "How many games to average results over.")
flags.DEFINE_integer(
    "eval_levels", 7,
    ("Play evaluation games vs MCTS+Solver, with max_simulations*10^(n/2)"
     " simulations for n in range(eval_levels). Default of 7 means "
     "running mcts with up to 1000 times more simulations."))
flags.DEFINE_integer("max_steps", 0, "How many learn steps before exiting.")
flags.DEFINE_bool("quiet", True, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS

EPISODE_LENGTH = 100

class TrajectoryState(object):
  """A particular point along a trajectory."""

  def __init__(self, observation, current_player, legals_mask, action, policy,
               value):
    self.observation = observation
    self.current_player = current_player
    self.legals_mask = legals_mask
    self.action = action
    self.policy = policy
    self.value = value


class Trajectory(object):
  """A sequence of observations, actions and policies, and the outcomes."""

  def __init__(self):
    self.states = []
    self.returns = None

  def add(self, information_state, action, policy):
    self.states.append((information_state, action, policy))


class Buffer(object):
  """A fixed size buffer that keeps the newest values."""

  def __init__(self, max_size):
    self.max_size = max_size
    self.data = []
    self.total_seen = 0  # The number of items that have passed through.

  def __len__(self):
    return len(self.data)

  def __bool__(self):
    return bool(self.data)

  def append(self, val):
    return self.extend([val])

  def extend(self, batch):
    batch = list(batch)
    self.total_seen += len(batch)
    self.data.extend(batch)
    self.data[:-self.max_size] = []

  def sample(self, count):
    return random.sample(self.data, count)


def _init_model_from_config(config):
  return model_lib.Model.build_model(
      config.nn_model,
      config.observation_shape,
      config.output_size,
      config.nn_width,
      config.nn_depth,
      config.weight_decay,
      config.learning_rate,
      config.path)

def _init_bot(config, game, evaluator_, evaluation):
  """Initializes a bot."""
  noise = None if evaluation else (config.policy_epsilon, config.policy_alpha)
  return mcts.MCTSBot(
      game,
      config.uct_c,
      config.max_simulations,
      evaluator_,
      solve=False,
      dirichlet_noise=noise,
      child_selection_fn=mcts.SearchNode.puct_value,
      verbose=False)

def _play_game(logger, game_num, game, bots, temperature, temperature_drop):
  """Play one game, return the trajectory."""
  trajectory = Trajectory()
  actions = []
  state = game.new_initial_state()
  logger.opt_print(" Starting game {} ".format(game_num).center(60, "-"))
  logger.opt_print("Initial state:\n{}".format(state))
  while not state.is_terminal():
    root = bots[state.current_player()].mcts_search(state)
    policy = np.zeros(game.num_distinct_actions())
    for c in root.children:
      policy[c.action] = c.explore_count
    policy = policy ** (1 / temperature)
    policy /= policy.sum()
    if len(actions) >= temperature_drop:
      action = root.best_child().action
    else:
      action = np.random.choice(len(policy), p=policy)
    trajectory.states.append(TrajectoryState(
        state.observation_tensor(), state.current_player(),
        state.legal_actions_mask(), action, policy,
        root.total_reward / root.explore_count))
    action_str = state.action_to_string(state.current_player(), action)
    actions.append(action_str)
    logger.opt_print("Player {} sampled action: {}".format(
        state.current_player(), action_str))
    state.apply_action(action)
  logger.opt_print("Next state:\n{}".format(state))

  trajectory.returns = state.returns()
  logger.print("Game {}: Returns: {}; Actions: {}".format(
      game_num, " ".join(map(str, trajectory.returns)), " ".join(actions)))
  return trajectory

def set_up(config):
  """Start all the worker processes for a full alphazero setup."""
  game = pyspiel.load_game(config.game)
  config = config._replace(
      observation_shape=game.observation_tensor_shape(),
      output_size=game.num_distinct_actions())

  print("Starting game", config.game)
  if game.num_players() != 2:
    sys.exit("AlphaZero can only handle 2-player games.")
  game_type = game.get_type()
  if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
    raise ValueError("Game must have terminal rewards.")
  if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
    raise ValueError("Game must have sequential turns.")
  if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
    raise ValueError("Game must be deterministic.")

  path = config.path
  if not path:
    path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
    config = config._replace(path=path)

  if not os.path.exists(path):
    os.makedirs(path)
  if not os.path.isdir(path):
    sys.exit("{} isn't a directory".format(path))
  print("Writing logs and checkpoints to:", path)
  print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                    config.nn_depth))

  with open(os.path.join(config.path, "config.json"), "w") as fp:
    fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

  return game, config

def save_to_reverb(trajectory):
  client = reverb.Client(f'localhost:12345')
  value = trajectory.returns[0]
  with client.trajectory_writer(num_keep_alive_refs=1) as writer:
    for s in trajectory.states:
      writer.append({
        'observation': np.array(s.observation, np.float32),
        'legals_mask': np.array(s.legals_mask, np.uint8),
        'policy': np.array(s.policy, np.float32),
        'value': np.array(value, np.float32),
      })
      writer.create_item(
        table='my_table',
        priority=1.5,
        trajectory={
          'observation': writer.history['observation'][-1:],
          'legals_mask': writer.history['legals_mask'][-1:],
          'policy': writer.history['policy'][-1:],
          'value': writer.history['value'][-1:],
        },
      )

def actor(config):
  """An actor process runner that generates games and returns trajectories."""
  logger = file_logger.FileLogger(config.path, "actor", config.quiet)
  game, config = set_up(config)
  model = _init_model_from_config(config)
  az_evaluator = evaluator_lib.AlphaZeroEvaluator(game, model)
  bots = [
      _init_bot(config, game, az_evaluator, False),
      _init_bot(config, game, az_evaluator, False),
  ]
  for game_num in itertools.count():
    trajectory = _play_game(logger, game_num, game, bots, config.temperature, config.temperature_drop)
    save_to_reverb(trajectory)

def main(_):
  config = alpha_zero.Config(
      game=FLAGS.game,
      path=FLAGS.path,
      learning_rate=FLAGS.learning_rate,
      weight_decay=FLAGS.weight_decay,
      train_batch_size=FLAGS.train_batch_size,
      replay_buffer_size=FLAGS.replay_buffer_size,
      replay_buffer_reuse=FLAGS.replay_buffer_reuse,
      max_steps=FLAGS.max_steps,
      checkpoint_freq=FLAGS.checkpoint_freq,
      actors=FLAGS.actors,
      evaluators=FLAGS.evaluators,
      uct_c=FLAGS.uct_c,
      max_simulations=FLAGS.max_simulations,
      policy_alpha=FLAGS.policy_alpha,
      policy_epsilon=FLAGS.policy_epsilon,
      temperature=FLAGS.temperature,
      temperature_drop=FLAGS.temperature_drop,
      evaluation_window=FLAGS.evaluation_window,
      eval_levels=FLAGS.eval_levels,
      nn_model=FLAGS.nn_model,
      nn_width=FLAGS.nn_width,
      nn_depth=FLAGS.nn_depth,
      observation_shape=None,
      output_size=None,
      quiet=FLAGS.quiet,
  )
  actor(config)


if __name__ == "__main__":
  app.run(main)