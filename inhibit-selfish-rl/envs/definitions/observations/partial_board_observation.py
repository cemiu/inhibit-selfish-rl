import numpy as np
from gym.vector.utils import spaces

from envs.definitions.observations import Observation

AGENT_INPUTS = [
    "player_a_pos_x",
    "player_a_pos_y",
    "player_b_pos_x",
    "player_b_pos_y",
    "player_b_locked",  # 1 if door is closed, 0 if door is open / not present
    # "player_a_reward",
    # "player_b_reward",
]

# observation feature will be a NxN matrix of the board state
# surrounding the agent
AGENT_VISION_SIZE = 5


class PartialBoardObservation(Observation):
    """Observation
    2d numpy array with 5x5 board observation area surrounding the agent
        + 5 elements of type int64.

    Contains the following (in order):
        - board state [flattened] (5 x 5)
        - player_a_pos_x / player_a_pos_y
        - player_b_pos_x / player_b_pos_y
        - player_b_locked (1 if door is closed, 0 if door is open / not present)
        # - player_a_reward
        # - player_b_reward
        """
    def __init__(self, board_dim):
        super().__init__()
        self.board_dim = board_dim

    def get_observation(self, board, is_player_a=True):
        """Returns a new observation of the environment."""
        if is_player_a:
            vision_matrix = board.board_submatrix(board.player_a_pos, 5)
        else:
            vision_matrix = board.board_submatrix(board.player_b_pos, 5)

        state_obs = np.array(
            board.player_a_pos
            + board.player_b_pos
            + (
                board.door_exists,
                # board.player_a_resources,
                # board.player_b_resources,
            )
        )

        return np.concatenate([vision_matrix, state_obs])

    def get_observation_space(self):
        return spaces.Box(low=-1, high=int(1e6),  # -1 to 1 million
                          shape=(AGENT_VISION_SIZE ** 2 + len(AGENT_INPUTS),), dtype=np.int64)
