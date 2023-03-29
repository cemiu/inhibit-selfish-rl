"""
Author: cemiu (Philipp B.)
"""
from markdown.util import deprecated

import numpy as np
from typing import Dict, Union

from stable_baselines3.common.type_aliases import GymObs, Gym26ResetReturn, GymStepReturn, Gym26StepReturn


class State:
    """Wrapper for gym step return value."""

    def __init__(
        self,
        *state: Union[GymObs, Gym26ResetReturn, GymStepReturn, Gym26StepReturn]
    ):
        self._observation: GymObs = None  # type: ignore
        self._reward: float = None  # type: ignore
        self._done: bool = False
        self._info: Dict = {}

        if state:
            self.update(*state)

    def __str__(self):
        if len(self._observation) < 10:
            return self._str_short()
        return (
            f"State(observation={self.obs}, reward={self.reward}, "
            f"done={self._done}, info={self._info})"
        )

    def _str_short(self):
        return (
            f"State(observation=[truncated, len={len(self._observation)}], reward={self.reward},"
            f"done={self._done}, info={self._info})"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def obs(self) -> np.ndarray:
        return self._observation

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def done(self) -> bool:
        return self._done

    @property
    def info(self) -> Dict:
        return self._info

    def update(
        self,
        *state: Union[GymObs, Gym26ResetReturn, GymStepReturn, Gym26StepReturn]
    ):
        # print(type(state), isinstance(state, tuple), len(state), state)
        if len(state) == 1 and isinstance(state[0], tuple) and len(state[0]) in {2, 4, 5}:
            state = state[0]

        if isinstance(state, tuple) and len(state) in {2, 4, 5}:
            if len(state) == 2:  # Gym26ResetReturn
                obs, info = state
                self._observation = obs
                self._info = info
            else:  # GymStepReturn or Gym26StepReturn
                obs, reward, terminated, *rest = state
                self._observation = obs
                self._reward = reward

                if len(rest) == 1:  # GymStepReturn
                    truncated = False
                    info = rest
                else:  # Gym26StepReturn
                    truncated, info = rest

                self._done = terminated or truncated
                self._info = info
        else:
            raise ValueError("Invalid state input")



@deprecated
class VectorState:
    """Extension of state wrapper, supporting vectorised environments."""

    def __init__(self, step: tuple = None, reset: bool = False, vectorised: bool = False):
        self._vectorised = vectorised
        self._vector_count = 1 if not vectorised else len(step)

        if not vectorised:
            self._states = [State(step, reset)]
        else:
            self._states = [State(s, reset) for s in step]

    @property
    def obs(self):
        return [s.obs for s in self._states]

    @property
    def reward(self):
        return [s.reward for s in self._states]

    @property
    def done(self):
        return [s.done for s in self._states]
