import inspect
from functools import partial
from typing import Callable, Optional, Any
import warnings

from stable_baselines3.common.base_class import BaseAlgorithm

from envs.board import TwoPlayerBoard
from envs.definitions.board_stripping import BoardStrippingConfig


class SecondaryAgentConfig:
    def __init__(
            self,
            secondary_agent_mode: bool,
            model: BaseAlgorithm = None,
            reward_function: Optional[Callable[[TwoPlayerBoard, Optional[Any]], float]] = None,
            board_stripping_config: Optional[BoardStrippingConfig] = None,
            epsilon_greedy: Optional[float] = None,
    ):
        """
        :param secondary_agent_mode: A flag to indicate if the environment is being used for the secondary agent. (bool)
        :param model: Model to be used for secondary agent, to run its policy. (model object)
        :param board_stripping_config: A configuration for board stripping. (BoardStrippingConfig)
        :param reward_function: A function defining the reward for the secondary agent. (Callable)
        :param epsilon_greedy: A probability value (0 to 1) for an action being random. (float)
        """
        self._secondary_agent_mode = secondary_agent_mode
        self._model = model

        self._board_stripping_config = board_stripping_config
        self._reward_function = reward_function

        self._epsilon_greedy = epsilon_greedy
        if self._epsilon_greedy is None:
            self._epsilon_greedy = 0.0

        assert self._secondary_agent_mode is not None, "secondary_agent_mode must be set."
        if self._secondary_agent_mode:
            assert self._reward_function is not None, "reward_function must be set when running secondary agent."

        # Reward functions may work on both players. This ensures that it set to work for the secondary agent.
        if self._reward_function is not None:
            sig = inspect.signature(self._reward_function)
            if 'is_a_player' in sig.parameters:
                if sig.parameters['is_a_player'].default:
                    warnings.warn(
                        "The second agent's reward function will be modified to work for secondary agent."
                    )
                    self._reward_function = partial(self._reward_function, is_a_player=False)

        if self._secondary_agent_mode and self._model is not None:
            warnings.warn(
                "Secondary agent mode is enabled, so the default model will be ignored."
            )

    @property
    def board_stripping_config(self):
        return self._board_stripping_config

    @property
    def reward_function(self):
        return self._reward_function

    @property
    def epsilon_greedy(self):
        return self._epsilon_greedy

    @property
    def model(self):
        return self._model

    @property
    def secondary_agent_mode(self):
        return self._secondary_agent_mode

    def merge(self, config):
        """
        Overwrites the values of this configuration with the values of the provided configuration.

        :param config: A SecondaryAgentConfig object to overwrite the current configuration.
        """
        if config.board_stripping_config is not None:
            self._board_stripping_config = config.board_stripping_config
        if config.reward_function is not None:
            self._reward_function = config.reward_function
        if config.epsilon_greedy is not None:
            self._epsilon_greedy = config.epsilon_greedy
        if config.default_model is not None:
            self._model = config.default_model
        if config.secondary_agent_mode is not None:
            self._secondary_agent_mode = config.secondary_agent_mode
