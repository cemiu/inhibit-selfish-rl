import cv2
import numpy as np

from envs.board import BOARD_VALUES, TwoPlayerBoard
from util.render_util import COLOUR_MAP_BGR

# render specs
MARGIN = 2  # margin between blocks in pixels
BLOCK_DIM = 60  # block dimensions in pixels
SMALL_FRAC = 0.5  # fraction of block dimensions for small objects

# colour map for board objects
COLOURS = {
    'wall': 'white',
    'empty': 'black',
    'player_a': 'softgreen',
    'player_b': 'softred',
    'player_a_resources': 'darkgreen',
    'player_b_resources': 'darkred',
    'shared_resources': 'lightblue',
    'door': 'violet',
    'out_of_bounds': 'white',
}

# board objects which are rendered smaller (SMALL_FRAC)
RENDER_SMALL = {
    'player_a_resources',
    'player_b_resources',
    'shared_resources',
}

BLOCK_WIDTH = BLOCK_DIM
BLOCK_HEIGHT = BLOCK_DIM

# convert strings to lookup values
COLOURS = {BOARD_VALUES[k]: COLOUR_MAP_BGR[v] for k, v in COLOURS.items()}

# precompute
RENDER_SMALL = {BOARD_VALUES[k] for k in RENDER_SMALL}
SMALL_DIM = int(SMALL_FRAC * BLOCK_DIM)
SMALL_OFFSET = (BLOCK_DIM - SMALL_DIM) // 2


class RenderableBoard(TwoPlayerBoard):
    """A rendered for a board state based on OpenCV."""
    def __init__(  # noqa
            self,
            dimension,
            start_grid=None,
            player_a=None,
            player_b=None,
            draw_outline=True,
            title='RL Game',
            fps=30,
    ):
        """ Initialise the board renderer.

        Args:
            :param dimension: the dimension of the board
            :param start_grid: the initial board state
            :param player_a: the initial position of player a
            :param player_b: the initial position of player b
            :param draw_outline: whether to draw an outline around the board (1/2 block width & height)
            :param title: the title of the window
            :param fps: the frames per second of the window
        """
        self.width = dimension
        self.height = dimension

        self.draw_outline = draw_outline
        self.extra_size = 2 * MARGIN + BLOCK_DIM if draw_outline else 0
        self.extra_offset = MARGIN // 2 + BLOCK_DIM // 2 if draw_outline else 0

        self.screen_width = (BLOCK_WIDTH + MARGIN) * self.width + MARGIN + self.extra_size
        self.screen_height = (BLOCK_HEIGHT + MARGIN) * self.height + MARGIN + self.extra_size

        self.image = np.ones((self.screen_height, self.screen_width, 3), np.uint8) * 255

        if start_grid is None:
            self.grid = np.zeros((self.width, self.height)).tolist()
        else:
            self.grid = start_grid

        if player_a is not None:
            self.grid[player_a[0]][player_a[1]] = 1

        if player_b is not None:
            self.grid[player_b[0]][player_b[1]] = 2

        self.title = title

        self.fps = fps
        self.delay = int(1000 / self.fps)

    def set_grid(self, grid, player_a=None, player_b=None):
        """ Set the board state to be rendered next."""
        self.grid = grid

        if player_a is not None:
            self.grid[player_a[0]][player_a[1]] = 1

        if player_b is not None:
            self.grid[player_b[0]][player_b[1]] = 2

    def update(self):
        self.image = np.ones((self.screen_height, self.screen_width, 3), np.uint8) * 255
        # self.image = np.zeros((self.screen_height, self.screen_width, 3), np.uint8)
        for row in range(self.width):
            for column in range(self.height):

                colour = COLOURS[self.grid[row][column]]
                render_small = self.grid[row][column] in RENDER_SMALL
                x = (MARGIN + BLOCK_WIDTH) * column + MARGIN + self.extra_offset
                y = (MARGIN + BLOCK_HEIGHT) * row + MARGIN + self.extra_offset

                cv2.rectangle(self.image, (x, y), (x + BLOCK_WIDTH, y + BLOCK_HEIGHT),
                              COLOURS[BOARD_VALUES['empty']] if render_small else colour, -1)

                if render_small:
                    x, y = x + SMALL_OFFSET, y + SMALL_OFFSET
                    x2, y2 = x + BLOCK_WIDTH - (2 * SMALL_OFFSET), y + BLOCK_WIDTH - (2 * SMALL_OFFSET)
                    cv2.rectangle(self.image, (x, y), (x2, y2), colour, -1)

        if self.draw_outline:
            square = self.width + 2
            row_top = [(0, i) for i in range(square)]
            row_bottom = [(square - 1, i) for i in range(square)]
            column_left = [(i, 0) for i in range(1, square - 1)]
            column_right = [(i, square - 1) for i in range(1, square - 1)]

            for r, c in row_top + row_bottom + column_left + column_right:
                x = (MARGIN + BLOCK_WIDTH) * c + MARGIN - self.extra_offset
                y = (MARGIN + BLOCK_HEIGHT) * r + MARGIN - self.extra_offset
                cv2.rectangle(self.image, (x, y), (x + BLOCK_WIDTH, y + BLOCK_HEIGHT),
                              COLOURS[BOARD_VALUES['out_of_bounds']], -1)

        cv2.imshow(self.title, self.image)
        cv2.waitKey(self.delay)

    # noinspection PyMethodMayBeStatic
    def close(self):
        cv2.destroyAllWindows()

