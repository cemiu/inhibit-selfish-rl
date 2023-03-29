import numpy as np

from util import matrix_utils

_movement_map = [(0, -1), (0, 1), (-1, 0), (1, 0)]
_movement_map_reverse = [(0, 1), (0, -1), (1, 0), (-1, 0)]

BOARD_VALUES = {
    "wall": -1,
    "empty": 0,
    "player_a": 1,
    "player_b": 2,
    "player_a_resources": 3,
    "player_b_resources": 4,
    "shared_resources": 5,
    "door": 6,
    "out_of_bounds": None,
}

_BOARD_LOOKUP = {v: k for k, v in BOARD_VALUES.items()}


class TwoPlayerBoard:
    """A board containing two players.

    The board is represented as a numpy array of shape (board_dimension, board_dimension).

    Each player has their own resources, and a shared resource.
    Doors can only be opened by player A, by contact.

    Invalid moves are permitted, but will not change the board state.
    Only player A's moves are counted as steps.

    Values:
        -1: Wall
        0: Empty
        # 1: Player A -- players are not represented in the board state anymore
        # 2: Player B -- players are not represented in the board state anymore
        3: Player A's resources
        4: Player B's resources
        5: Shared resources
        6: Door
    """
    def __init__(
            self,
            board_dimension,
            player_a_pos=None,
            player_b_pos=None,
            board_state=None,
            max_steps=1000,
    ):
        """Initialise the board.

        Args:
            :param board_dimension: The dimension of the board.
            :param player_a_pos: The initial position of player A.
            :param player_b_pos: The initial position of player B.
            :param board_state: The initial state of the board.
            :param max_steps: The maximum number of steps before the game ends.
            # :param reward_generator: A function that places rewards on the board.
        """
        self.board_dimension = board_dimension
        self.player_a_pos = player_a_pos
        self.player_b_pos = player_b_pos

        self.player_a_resources = 0
        self.player_b_resources = 0

        # has collected/opened on last step?
        self.has_a_collected_resource = False
        self.has_b_collected_resource = False
        self.has_a_opened_door = False

        self.board_state = board_state
        self.max_steps = max_steps
        self.step_count_player_a = 0
        self.step_count_player_b = 0
        self.remaining_steps = self.max_steps - self.step_count_player_a

        self.last_action_a = None
        self.last_action_b = None

        self.together_count = 1 * self.are_a_and_b_together  # 1 if they are together, 0 otherwise
        self.door_open_step = None  # the step at which the door was opened
        self.a_rew_collect_step = None  # the step at which player A collected all resources
        self.b_rew_collect_step = None  # the step at which player B collected all resources
        self.a_touches_b = False  # True on steps on which player A touches player B, but has not before

        # the following two states are not used, as constant time penalty-based reward is used
        self.invalid_move_count_player_a = 0
        self.last_reward_step_player_a = 0

        if self.board_state is None:
            self.board_state = np.zeros((self.board_dimension, self.board_dimension), dtype=np.int8)

        if self.player_a_pos is None:
            self.player_a_pos = (0, 0)

        if self.player_b_pos is None:
            self.player_b_pos = (self.board_dimension - 1, self.board_dimension - 1)

        self.door_exists = 6 in self.board_state
        self.remaining_a_resources = np.count_nonzero(self.board_state == 3) + np.count_nonzero(self.board_state == 5)
        self.remaining_b_resources = np.count_nonzero(self.board_state == 4) + np.count_nonzero(self.board_state == 5)

    @property
    def state(self):
        """Return the board state as a flattened numpy array.

        The board state is flattened, and the player positions are added,
        potentially obfuscating the board state.
        """
        flattened = self.board_state.flatten()
        flattened[self.player_a_pos[0] * self.board_dimension + self.player_a_pos[1]] = 1
        flattened[self.player_b_pos[0] * self.board_dimension + self.player_b_pos[1]] = 2

        return flattened

    @property
    def time_waste_count_player_a(self):
        """Return the number of steps since the last reward was collected for player A.

        It is not used, as constant time penalty-based reward is used instead."""
        return self.step_count_player_a - self.last_reward_step_player_a

    @property
    def are_a_resources_left(self):
        """Return whether there are any resources left on the board for player A."""
        return self.remaining_a_resources > 0  # Only consider player A's resources

    @property
    def are_b_resources_left(self):
        """Return whether there are any resources left on the board for player B."""
        return self.remaining_b_resources > 0

    @property
    def are_a_and_b_together(self):
        """Return whether player A and player B are on the same tile."""
        return self.player_a_pos == self.player_b_pos

    def move_player_a(self, action, dont_move=False):
        """Move player A in the given direction.

        Args:
            :param action: 0 = left, 1 = right, 2 = up, 3 = down
            :param dont_move: If True, the player will not move, but the board state will be updated.
        """
        assert action in [0, 1, 2, 3], "Invalid action for moving player A: {}".format(action)
        self.has_a_collected_resource = False
        self.has_a_opened_door = False

        pre_move_together = self.are_a_and_b_together

        if not dont_move:
            # maps action to a movement vector and adds it to the player's position
            new_pos = tuple(map(sum, zip(self.player_a_pos, _movement_map[action])))
            if self._is_valid_move(new_pos, is_player_a=True):
                self.player_a_pos = new_pos

                match self.board_state[self.player_a_pos]:
                    case 3:  # player A's resources
                        self.board_state[self.player_a_pos] = 0
                        self.player_a_resources += 1
                        self.remaining_a_resources -= 1
                        self.last_reward_step_player_a = self.step_count_player_a
                        self.has_a_collected_resource = True
                        if self.remaining_a_resources == 0:
                            self.a_rew_collect_step = self.step_count_player_a + 1
                    case 5:  # shared resources
                        self.board_state[self.player_a_pos] = 0
                        self.player_a_resources += 1
                        self.remaining_a_resources -= 1
                        self.remaining_b_resources -= 1
                        self.last_reward_step_player_a = self.step_count_player_a
                        self.has_a_collected_resource = True

                        if self.remaining_a_resources == 0:
                            self.a_rew_collect_step = self.step_count_player_a + 1
                        if self.remaining_b_resources == 0:
                            self.b_rew_collect_step = self.step_count_player_b
                    case 6:  # remove door by walking into it, and reset position
                        self.board_state[self.player_a_pos] = 0
                        self.door_exists = 6 in self.board_state
                        self.player_a_pos = tuple(map(sum, zip(self.player_a_pos, _movement_map_reverse[action])))
                        # self.last_reward_step_player_a = self.step_count_player_a
                        self.has_a_opened_door = True
                        self.door_open_step = self.step_count_player_a + 1
                    case _:
                        pass
            else:
                # invalid move
                self.invalid_move_count_player_a += 1

        self.step_count_player_a += 1
        self.a_touches_b = False
        if self.are_a_and_b_together and not pre_move_together:
            self.together_count += 1
            self.a_touches_b = True

        if isinstance(action, np.ndarray):
            action = action.item()
        self.last_action_a = action

    def move_player_b(self, action, dont_move=False):
        """Move player B in the given direction.

        Args:
            action: 0 = left, 1 = right, 2 = up, 3 = down
            dont_move: if True, the player will not move, but the board will be updated as if it had moved
        """
        assert action in [0, 1, 2, 3], "Invalid action for moving player B: {}".format(action)
        self.has_b_collected_resource = False

        pre_move_together = self.are_a_and_b_together

        # maps action to a movement vector and adds it to the player's position
        if not dont_move:
            new_pos = tuple(map(sum, zip(self.player_b_pos, _movement_map[action])))
            if self._is_valid_move(new_pos, is_player_a=False):
                self.player_b_pos = new_pos
                match self.board_state[self.player_b_pos]:
                    case 4:  # player B's resources
                        self.board_state[self.player_b_pos] = 0
                        self.player_b_resources += 1
                        self.remaining_b_resources -= 1
                        self.has_b_collected_resource = True

                        if self.remaining_b_resources == 0:
                            self.b_rew_collect_step = self.step_count_player_b + 1
                    case 5:  # shared resources
                        self.board_state[self.player_b_pos] = 0
                        self.player_b_resources += 1
                        self.remaining_a_resources -= 1
                        self.remaining_b_resources -= 1
                        self.has_b_collected_resource = True

                        if self.remaining_a_resources == 0:
                            self.a_rew_collect_step = self.step_count_player_a
                        if self.remaining_b_resources == 0:
                            self.b_rew_collect_step = self.step_count_player_b + 1
                    case _:
                        pass

        self.step_count_player_b += 1
        self.remaining_steps = self.max_steps - self.step_count_player_b
        if self.are_a_and_b_together and not pre_move_together:
            self.together_count += 1

        if isinstance(action, np.ndarray):
            action = action.item()
        self.last_action_b = action

    def _is_valid_move(self, pos, is_player_a=False) -> bool:
        """Check if a move is valid, given the board state and the player's position.

        Player A can walk into walls to open them, but Player B cannot.
        """
        if pos[0] < 0 or pos[0] >= self.board_dimension or pos[1] < 0 or pos[1] >= self.board_dimension:
            return False  # out of bounds
        if self.board_state[pos] == -1:
            return False  # wall
        if not is_player_a and self.board_state[pos] == BOARD_VALUES['door']:
            return False  # B can't open doors
        return True

    def board_submatrix(
            self,
            pos: tuple[int, int],
            size: int,
            include_agents: bool = False
    ) -> np.ndarray:
        """Return a submatrix of the board, centered on a given position.

        Args:
            pos: The position to center the submatrix on.
            size: The size of the submatrix. Must be odd, else it will be rounded up.
            include_agents: Whether to include the agents in the submatrix.

        Returns:
            The submatrix.
        """
        size = size if size % 2 else size + 1
        radius = size // 2
        sub_matrix = matrix_utils.extract_sub_matrix(self.board_state, pos, radius)

        if include_agents:
            def update_submatrix_with_agent(agent_pos: tuple[int, int], agent_value: int):
                row, col = agent_pos
                if matrix_utils.is_inside_submatrix(pos, row, col, radius):
                    sub_matrix[row - pos[0] + radius, col - pos[1] + radius] = agent_value

            update_submatrix_with_agent(self.player_a_pos, BOARD_VALUES['player_a'])
            update_submatrix_with_agent(self.player_b_pos, BOARD_VALUES['player_b'])

        return sub_matrix.flatten()
