from typing import Optional

import gym
import numpy as np

from ml.reward_networks import LoadableRewardNet


class RewardModel(LoadableRewardNet):
    """
    A reward model that can predict rewards given a state, action, and next state.

    This class is a wrapper around the LoadableRewardNet class,
    but has the ability to handle non-vectorised inputs.

    # stacked_single = th.stack([inputs_concat[0]])
    """
    def __init__(
            self,
            venv: Optional[gym.Env],
            observation_space: Optional[gym.spaces.Space] = None,
            action_space: Optional[gym.spaces.Space] = None,
            initial_state: np.ndarray | None = None,
            weight: float = 1.0,
    ):
        assert venv is not None or (observation_space is not None and action_space is not None), \
            "Either venv or observation_space and action_space must be provided."

        if venv is not None:
            observation_space = venv.observation_space
            action_space = venv.action_space
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            use_state=True,
            use_action=True,
            use_next_state=True,
            use_done=False,
        )

        self.last_state = None
        self.weight = weight

    def reset(
            self,
            state: np.ndarray | None = None,
    ) -> None:
        """Resets the reward model."""
        self.last_state = state

    def set_initial_state(
            self,
            state: np.ndarray | None = None,
    ) -> None:
        """Sets the initial state of the reward model."""
        self.reset(state)

    # noinspection PyMethodOverriding
    def predict(
        self,
        action: np.ndarray,
        state: np.ndarray,
    ) -> float:
        """Predicts the reward for a given action and the resulting state."""
        if self.last_state is None:
            self.last_state = state
            import warnings
            warnings.warn("Reward model asked for prediction before initial state was set. Returning reward: 0.0.")
            return 0.0

        predicted_reward = self._predict(
            state=self.last_state,
            action=action,
            next_state=state,
        )

        self.last_state = state
        return predicted_reward

    def _predict(
            self,
            state: np.ndarray,
            action: np.ndarray | int,
            next_state: np.ndarray | None = None,
            done: np.ndarray | bool = False
    ) -> np.ndarray | float:
        """Overwrites the predict method to handle non-vectorised inputs."""
        is_vector = isinstance(action, np.ndarray)
        if not is_vector:
            state = np.stack([state])
            action = np.stack([action])
            next_state = np.stack([next_state])
            done = np.stack([done])

        prediction = super().predict(state, action, next_state, done)
        if is_vector:
            return prediction
        return prediction[0]

    @classmethod
    def load(
            cls,
            path,
            venv: Optional[gym.Env] = None,
            observation_space: Optional[gym.spaces.Space] = None,
            action_space: Optional[gym.spaces.Space] = None,
            weight: Optional[float] = 1.0,
    ) -> LoadableRewardNet:
        assert venv is not None or (observation_space is not None and action_space is not None), \
            "Either venv or observation_space and action_space must be provided."
        network = cls(
            venv=venv,
            observation_space=observation_space,
            action_space=action_space,
            weight=weight,
        )
        network.setnet(path)
        return network
