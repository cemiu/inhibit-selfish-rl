""" A wrapper combining A's internal reward function
with B's learned reward model.

The purpose is to let A internalise B's reward model,
and be considerate of B's preferences.
"""
from collections.abc import Callable
from functools import partial
from typing import Optional, Union

import gym
import numpy as np

from envs.board import TwoPlayerBoard
from ml.reward_model import RewardModel


class RewardWrapper:
    """ A wrapper combining A's internal reward function
    with B's learned reward model."""
    def __init__(
            self,
            a_reward_function: Callable,
            a_observation_function: Callable,
            b_reward_model: Optional[Union[RewardModel, str]] = None,
            b_reward_model_weight: float = 1.0,
            b_observation_function: Optional[Callable] = None,
            venv: Optional[gym.Env] = None,
            observation_space: Optional[gym.spaces.Space] = None,
            action_space: Optional[gym.spaces.Space] = None,
    ):
        """ Initialise the combined reward wrapper.
        If no reward model is provided, the wrapper will simply return A's reward function.

        Args:
            :param a_reward_function: A's objective reward function, defining self-interest
            :param a_observation_function: A's observation function (Observation.get_observation)
            :param b_reward_model: B's learned reward model, defining B's preferences,
                as a RewardModel object or a path to a saved model
            :param b_observation_function: B's observation function (Observation.get_observation)
            :param venv: The vectorised environment, used to extract the observation and action spaces
            :param observation_space: The observation space of the environment
            :param action_space: The action space of the environment
        """
        self.a_reward_function = a_reward_function
        self.b_reward_model = b_reward_model
        self.b_reward_model_weight = b_reward_model_weight

        self.a_observation_function = a_observation_function
        self.b_observation_function = b_observation_function

        if b_reward_model is not None:
            assert b_observation_function is not None, \
                "If a reward model for B is provided, an observation function must also be provided."

        if isinstance(self.b_reward_model, str):
            assert venv or (observation_space and action_space), \
                "Either an environment or observation space and action space must be provided, " \
                "to instantiate a reward model."

            self.b_reward_model = RewardModel.load(
                path=b_reward_model,
                venv=venv,
                observation_space=observation_space,
                action_space=action_space,
                weight=b_reward_model_weight,
            )

    def reset(
            self,
            board: TwoPlayerBoard,
    ) -> None:
        """Resets the reward model, with the initial observation."""
        assert board.step_count_player_a == 0, \
            "The reward wrapper should start with a fresh board."

        if self.b_reward_model is None:
            return

        state = self.b_observation_function(board)
        self.b_reward_model.reset(state)

    def evaluate(
            self,
            board: TwoPlayerBoard,
            is_player_a: bool = True,
    ) -> float:
        """This method evaluates the reward function and mode, and returns the combined reward.

        It should be called on every step of the environment, to get the returned reward."""
        assert is_player_a, "Applying the reward wrapper to B is not supported, " \
                            "and would produce nonsensical results."

        # The normal, internal reward function
        internal_reward = self.a_reward_function(board)

        if self.b_reward_model is None:
            return internal_reward

        action = board.last_action_b
        resulting_state = self.b_observation_function(board)
        external_reward = self.b_reward_model.predict(action=action, state=resulting_state)

        # no additional weighting is needed
        internalised_reward = internal_reward + external_reward * self.b_reward_model.weight
        return internalised_reward


class RewardWrapperConfig:
    """A configuration for the combined reward wrapper.

    This is passed to an environment, which then self-configures
    the reward wrapper.
    """
    def __init__(
            self,
            a_reward_function: Callable,
            b_reward_model_path: str = None,
            b_reward_model_weight: float = 1.0,
    ):
        """Initialise the combined reward wrapper configuration.

        Args:
            :param a_reward_function: A's objective reward function, defining self-interest
            :param b_reward_model_path: B's learned reward model, defining B's preferences,
        """
        self.a_reward_function = a_reward_function
        self.b_reward_model_path = b_reward_model_path
        self.b_reward_model_weight = b_reward_model_weight

    def get_reward_wrapper(
            self,
            action_space: gym.spaces.Space,
            observation_space: gym.spaces.Space,
            observation_function: Callable[[TwoPlayerBoard, Optional[bool]], np.ndarray],
            # observation_function_a: Callable[[TwoPlayerBoard, Optional[bool]], np.ndarray],
            # observation_function_b: Callable[[TwoPlayerBoard, Optional[bool]], np.ndarray],
    ) -> RewardWrapper:
        """Returns the reward wrapper configured with this configuration.

        Args:
            :param action_space: The action space of the environment
            :param observation_space: The observation space of the environment
            :param observation_function: Agent observation function (Observation.get_observation)
        """
        # pre-fill player identities
        observation_function_a = partial(observation_function, is_player_a=True)
        observation_function_b = partial(observation_function, is_player_a=False)

        return RewardWrapper(
            a_reward_function=self.a_reward_function,
            b_reward_model=self.b_reward_model_path,
            b_reward_model_weight=self.b_reward_model_weight,
            observation_space=observation_space,
            action_space=action_space,
            a_observation_function=observation_function_a,
            b_observation_function=observation_function_b,
        )