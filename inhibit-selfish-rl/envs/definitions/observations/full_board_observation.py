import numpy as np
from gym.vector.utils import spaces

from envs.definitions.observations import Observation

AGENT_INPUTS = [
    "player_a_pos_x",
    "player_a_pos_y",
    "player_b_pos_x",
    "player_b_pos_y",
    "player_a_reward",
    "player_b_reward",
    "player_b_locked",  # 1 if door is closed, 0 if door is open / not present
    "player_a_reward_remaining",
    "player_b_reward_remaining",
    "player_a_steps_remaining",
]


class FullBoardObservation(Observation):
    """Includes the entire board state and many .other params.
    2d numpy array with BOARD_DIMENSION^2 + 10 elements of type int64.

    Contains the following (in order):
        - player_a_pos_x / player_a_pos_y
        - player_b_pos_x / player_b_pos_y
        - player_a_reward
        - player_b_reward
        - player_b_locked (1 if door is closed, 0 if door is open / not present)
        - player_a_reward_remaining
        - player_b_reward_remaining
        - player_a_steps_remaining
        - board state [flattened] (BOARD_DIMENSION x BOARD_DIMENSION)
        """
    def __init__(self, board_dim):
        super().__init__()
        self.board_dim = board_dim

    def get_observation(self, board, is_player_a=True):
        """Returns a new observation of the environment."""
        state_obs = np.array(
            board.player_a_pos
            + board.player_b_pos
            + (board.player_a_resources,
               board.player_b_resources,
               board.door_exists,
               board.remaining_a_resources,
               board.remaining_b_resources,
               board.remaining_steps
               )
        )

        return np.concatenate([state_obs, board.state])

    def get_observation_space(self):
        return spaces.Box(low=-1, high=int(1e6),  # -1 to 1 million
                          shape=(len(AGENT_INPUTS) + self.board_dim ** 2,), dtype=np.int64)
