import numpy as np
from gym.vector.utils import spaces

from envs.board import TwoPlayerBoard
from util.python_utils import ForceFunctionMeta, forcefunction


class Observation(metaclass=ForceFunctionMeta):
    """Abstract class for observations.

    Must implement get_observation_space and get_observation.
    """
    def __init__(self, *_, **__):
        pass

    @forcefunction
    def get_observation_space(self) -> spaces.Box:
        pass

    @forcefunction
    def get_observation(self, board: TwoPlayerBoard, is_player_a: bool = True) -> np.ndarray:
        pass
