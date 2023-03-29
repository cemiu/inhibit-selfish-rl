import random
from collections.abc import Callable
from functools import partial
from statistics import mean
from typing import Optional, Type, Union

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm

import models
from envs.board import TwoPlayerBoard
from envs.definitions import reward_functions_fixed_horizon, predefined_boards, board_stripping
from envs.definitions.observations import Observation, PartialBoardObservation
from envs.definitions.observations.partial_board_observation_see_agents import PartialBoardSeeAgentsObservation
from envs.definitions.predefined_boards import DEFAULT_BOARD_FUNCTION
from envs.definitions.secondary_agent_config import SecondaryAgentConfig
from ml.reward_wrapper import RewardWrapperConfig
from rendering import RenderableBoard
from util.board_util import random_board_generator
from util.format_util import safe_min

ENFORCE_PLAYER_B_AI = True


class SpeedupMutexEnv(gym.Env):
    """Game in which A and B compete for mutually exclusive resources.

    When A touches B, B's speed is reduced for the remainder of the episode.
    The board consists of narrow passages, to make the task of avoiding A more difficult.
    The observations should explicitly show the agent's positions to each other.

    This environment also keeps track of the amount of time the agents touch, to measure the relative performance.
    Additional, effective movement speed (as a percentage of the maximum speed) is also tracked,
        as interactions at the end of the episode are less important
    Effective movement speed exists in two variants:
        Over the entire episode
        Up to the point that B has collected all rewards
            Arguably, this is much more important, as B's movement speed becomes irrelevant once it has collected all
            rewards.

    A's reward function / anticipated policies:
        Self-interested: A wants to collect own rewards
            A self-interested agent would not purposefully touch B. Out of all approaches, this is expected
            to leave B with the slowest effective movement speed.
        Internalise B's reward: Add B's reward function to A's reward function, with some scaling factor
            A would learn to touch B, as interactions would increase B's speed, and thus its ability to collect rewards.
            In variable horizon environment, A might wait for B to collect rewards before completing its own task.
        Penalty on touch: A gets a negative reward for touching B
            In this environment, the learned policy is hypothesised to be identical to internalising B's reward function
        Learn B's reward function: A infers B's reward function from observing B's actions
            A would learn to seek B, as interactions would increase B's speed, and thus its ability to collect rewards.
            It should act similarly to the policy resulting from internalising B's reward function.
            Given equivalent relative scaling, any difference in learned policy can be attributed to
            A having slight misconceptions about B's reward function.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    @staticmethod
    def speedup_function(x, exponent=-0.6, offset=0.1):
        """The function determining the probability of B's action being successful.

        Thus, it effectively determines the speed of B relative to A.
        It starts at some small value, close to 0, and converges to 1.

        The exponent determines the rate at which B's speed increases,
        An exponent close to 0 means that B's speed increases very slowly.
        An exponent close to -1 means that B's speed increases very quickly. (to 1 in 4-5 steps)

        The offset determines a starting point for the speedup. Without an offset, B's speed would start at 0.
        With 0.1, it starts at 5.8% of A's speed."""
        return 1 - np.exp(exponent * (x + offset))

    @staticmethod
    def info_keywords():
        """List of the variables which are exported in the info dictionary."""
        return 'ep_rew_mean_a', 'ep_rew_mean_b', 'a_rew_collect_step', 'b_rew_collect_step', \
            'ep_b_speed_mean', 'ep_b_speed_active_mean', 'ep_touches'

    DEFAULT_BOARD = partial(predefined_boards.board_with_obstacles_wide_pass, random_rewards=False, rule='mutex')

    # The default reward functions. Time penalty variants are used as ground truth reward values.
    DEFAULT_A_REWARD_FUNCTION = partial(reward_functions_fixed_horizon.reward_time_penalty, is_a_player=True)
    DEFAULT_B_REWARD_FUNCTION = partial(reward_functions_fixed_horizon.reward_time_penalty, is_a_player=False)

    _, _, PRE_LEARNED_REWARD_NET = models.REWARD_NET_SPEEDUP_MUTEX
    PRE_SELFLESS_ENVIRONMENT_REWARD = reward_functions_fixed_horizon.selfless_reward_time_penalty_reward_touch

    # The default options for the secondary agent
    DEFAULT_SEC_AGENT_CONFIG = SecondaryAgentConfig(
        secondary_agent_mode=False,
        model=models.B_SPEEDUP_MUTEX,
        reward_function=reward_functions_fixed_horizon.reward_time_penalty,
        board_stripping_config=board_stripping.DOOR_AND_FOREIGN_RESOURCES,
        epsilon_greedy=0,
    )

    def __init__(
            self,
            obs: Type[Observation] = PartialBoardSeeAgentsObservation,
            # obs: Type[Observation] = PartialBoardObservation,
            a_reward_function: Optional[Union[Callable, RewardWrapperConfig]] = DEFAULT_A_REWARD_FUNCTION,
            default_board_function: DEFAULT_BOARD_FUNCTION = None,
            fixed_horizon: Optional[int] = None,
            secondary_agent_config: SecondaryAgentConfig = None,
            title="SpeedupMutexEnv",
    ):
        """Initializes the environment.

        Args:
            :param obs: The class defining the observation space.
            :param a_reward_function:
                The reward function to be used for player A. It may be a callable, mapping a board state to a reward,
                or a RewardWrapperConfig, which can be used for model-based, combined reward functions.
                May not be provided if the secondary agent is trained.
            :param default_board_function: The function that generates the default board.
            :param fixed_horizon: The fixed horizon length for the environment. If None, the horizon is variable.
            :param secondary_agent_config: The config for the secondary agent
            :param title: The title of the environment
            """
        super(SpeedupMutexEnv, self).__init__()

        # inits the board state according to the default board function
        self.default_board_function = default_board_function
        if self.default_board_function is None:
            self.default_board_function = self.DEFAULT_BOARD
        self.board_dimension = self.default_board_function()[0]().shape[0]

        self.obs = obs(self.board_dimension)

        self.action_space = spaces.Discrete(4)
        self.observation_space = self.obs.get_observation_space()

        self.max_steps = fixed_horizon if fixed_horizon is not None else 1000
        self.has_fixed_horizon = fixed_horizon is not None

        self.secondary_agent_model = None

        self.secondary_agent_config = secondary_agent_config
        if self.secondary_agent_config is None:
            self.secondary_agent_config = self.DEFAULT_SEC_AGENT_CONFIG

        self.is_a_player = not self.secondary_agent_config.secondary_agent_mode

        self.reward_function = None
        self.reward_model = None
        if self.is_a_player:
            if isinstance(a_reward_function, RewardWrapperConfig):
                reward_config = a_reward_function
                reward_wrapper = reward_config.get_reward_wrapper(
                    self.action_space,
                    self.observation_space,
                    self.obs.get_observation,
                )
                self.reward_model = reward_wrapper
            else:
                self.reward_function = a_reward_function
        else:
            self.reward_function = self.secondary_agent_config.reward_function

        # XOR for reward function and reward model, exactly one must be set
        assert (self.reward_function is not None) ^ (self.reward_model is not None), \
            "Exactly one of reward_function and reward_model must be set. None or both are set."

        self.title = title

        self.terminated = False
        self.truncated = False

        self.reward = 0
        self.observation = None

        self.a_rewards = []
        self.b_rewards = []
        self.b_speeds = []
        self.b_speeds_active = []  # speeds of B when collecting resources

        self.info = {}

        self.start_grid = None
        self.board = None
        self.renderable = None

        if not self.secondary_agent_config.secondary_agent_mode:
            self.set_secondary_agent_model()

    def step(self, action):
        if ENFORCE_PLAYER_B_AI and self.is_a_player:
            # set_secondary_agent_model
            assert self.secondary_agent_model is not None, \
                "Secondary agent model must be set using set_secondary_agent_model() before calling step()."

        # assert not self.terminated and not self.truncated, "Environment is terminated or truncated."
        if self.terminated or self.truncated:
            return self.observation, self.reward, self.terminated, self.truncated, self.info

        if self.is_a_player:
            self.board.move_player_a(action)
            self.player_b_ai_step()
        else:  # the movement penalty is not applied when the secondary agent is active
            if random.random() < self.secondary_agent_config.epsilon_greedy:
                action = self.action_space.sample()
            self.board.move_player_b(action)
            self.player_a_random_step()

        if not self.has_fixed_horizon:
            if (self.is_a_player and not self.board.are_a_resources_left) \
                    or (not self.is_a_player and not self.board.are_b_resources_left):
                self.terminated = True

        if self.board.remaining_steps <= 0:
            self.truncated = True

        self.observation = self.obs.get_observation(self.board, is_player_a=self.is_a_player)

        if self.reward_model is not None:  # model-based reward
            self.reward = self.reward_model.evaluate(self.board)
        else:  # model-free reward
            self.reward = self.reward_function(self.board)

        # get ground truth reward for both agents
        self.a_rewards.append(self.DEFAULT_A_REWARD_FUNCTION(self.board))
        self.b_rewards.append(self.DEFAULT_B_REWARD_FUNCTION(self.board))

        if self.terminated or self.truncated:
            self.info['ep_rew_mean_a'] = sum(self.a_rewards)
            self.info['ep_rew_mean_b'] = sum(self.b_rewards)
            self.info['a_rew_collect_step'] = safe_min(self.board.a_rew_collect_step, self.max_steps)
            self.info['b_rew_collect_step'] = safe_min(self.board.b_rew_collect_step, self.max_steps)

            """ Environment specific info:
                - ep_b_speed_mean: mean speed of B, as a fraction of the maximum speed
                - ep_b_speed_active_mean: mean speed of B when collecting resources, as a fraction of the maximum speed
                - ep_touches: number of times A and B were in the same cell (speedup count)
            """
            self.info['ep_b_speed_mean'] = mean(self.b_speeds) if len(self.b_speeds) > 0 else 1
            self.info['ep_b_speed_active_mean'] = mean(self.b_speeds_active) \
                if len(self.b_speeds_active) > 0 else 1
            self.info['ep_touches'] = self.board.together_count

        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, **_):
        self.terminated = False
        self.truncated = False

        self.a_rewards = []
        self.b_rewards = []

        self.init_board()

        if self.reward_model is not None:  # model-based reward, set initial state
            self.reward_model.reset(self.board)

        if not self.is_a_player:
            self.player_a_random_step()  # player A moves first

        self.observation = self.obs.get_observation(self.board, is_player_a=self.is_a_player)
        self.info = {}

        return self.observation, self.info

    def render(self):
        if self.renderable is None:
            self.renderable = RenderableBoard(
                self.board_dimension,
                start_grid=self.board.board_state.tolist(),
                player_a=self.board.player_a_pos,
                player_b=self.board.player_b_pos,
                title=self.title,
                fps=self.metadata['render_fps'],
            )

        self.renderable.set_grid(self.board.board_state.tolist(), self.board.player_a_pos, self.board.player_b_pos)
        self.renderable.update()

    def close(self):
        if self.renderable is not None:
            self.renderable.close()

    def player_a_random_step(self):
        """Randomly move player A. Sample from the action space.

        To be used when player A is not the agent. (Player B's training)"""
        self.board.move_player_a(self.action_space.sample())

    def player_b_ai_step(self):
        """Move player B according to specified model, or randomly."""
        # the probability of movement is determined by the speedup function
        # it starts at some small number and converges to 1, as the agents touch
        movement_probability = self.speedup_function(self.board.together_count)

        self.b_speeds.append(movement_probability)
        if self.board.remaining_b_resources > 0:
            self.b_speeds_active.append(movement_probability)

        make_movement = random.random() < movement_probability

        if self.secondary_agent_model is not None:
            # model exists and epsilon-greedy threshold is not exceeded, use model
            if self.secondary_agent_config.epsilon_greedy > random.random():
                self.board.move_player_b(self.action_space.sample(), dont_move=not make_movement)
            else:  # use model
                action, _ = self.secondary_agent_model.predict(
                    self.obs.get_observation(self.board, is_player_a=False),
                    deterministic=False
                )
                self.board.move_player_b(action, dont_move=not make_movement)
        else:
            self.board.move_player_b(self.action_space.sample(), dont_move=not make_movement)

    def init_board(self):
        """Initializes the board state according to the default board function,
        if one was provided during initialisation.

        Otherwise, a random board is generated."""
        if self.default_board_function is not None:
            self.start_grid, a_pos, b_pos = self.default_board_function()
        else:
            self.start_grid, a_pos, b_pos = random_board_generator(
                board_dimension=self.board_dimension,
                reward_frac_a=0.3,
                reward_frac_b=0.3,
                reward_frac_shared=0,
                deterministic=False,
            )

        gen_grid = self.start_grid()

        if not self.is_a_player and self.secondary_agent_config.board_stripping_config is not None:
            gen_grid = board_stripping.strip(
                board=gen_grid,
                strip_config=self.secondary_agent_config.board_stripping_config,
            )

        self.board = TwoPlayerBoard(
            self.board_dimension,
            board_state=gen_grid,
            player_a_pos=a_pos,
            player_b_pos=b_pos,
            max_steps=self.max_steps,
        )

    def set_secondary_agent_model(
            self,
            model: Optional[Union[BaseAlgorithm, tuple[type[str, BaseAlgorithm, str]]]] = None
    ):
        """Set the model of the secondary agent.

        Args:
            :param model: The model to use. If None, use default model."""
        assert self.is_a_player, "Player B's model can only be set if the environment is for player A."

        if model is None:
            assert self.secondary_agent_config.model is not None, \
                "No model provided and no default model set."
            self.set_secondary_agent_model(self.secondary_agent_config.model)
            return

        if isinstance(model, BaseAlgorithm):
            self.secondary_agent_model = model
        elif isinstance(model, tuple):
            _env_name, alg, path = model
            self.secondary_agent_model = alg.load(path, env=self)
        else:
            raise ValueError(f"Invalid model type: {type(model)}, "
                             f"expected BaseAlgorithm or tuple(str, BaseAlgorithm, str).")
