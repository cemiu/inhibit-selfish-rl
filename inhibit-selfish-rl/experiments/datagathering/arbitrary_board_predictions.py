"""
This script is used to test the predictions of a model on an arbitrary board.

In particular, it is good at demonstrating how rewards placed on the board affect the
predictions of the model.

Some interesting path finding behaviours can be observed,
like:
 - preferring rewards which are chunked together
 - preferring rewards which are closer to the agent

"""

import numpy as np

import models


def square_string_to_array(board_str):
    rows = board_str.split("\n")
    rows = map(str.strip, rows)  # remove leading and trailing whitespace
    rows = map(lambda row: None if row.startswith('#') else row, rows)  # remove comments
    rows = list(filter(None, rows))  # remove empty rows

    # split each row into a list of ints
    data = list(map(lambda row: list(map(int, row.split())), rows))

    arr = np.array(data)
    print(arr)
    print()
    arr = arr.flatten()

    return arr


def generate_observation(board):
    """Generate an observation from the input string.

    Args:
        board (str): The input string.

    Returns:
        np.array: The observation.
    """
    board = square_string_to_array(board)
    misc = np.array([0, 0, 0, 0, 0])

    return np.concatenate((board, misc))


def test_prediction_for_board(board):
    """Test the prediction for a given board.

    Args:
        board (str): The board to test.
    """
    obs = generate_observation(board)

    alg, model = models.DOOR_MUTEX_10x10_B_1M
    model = alg.load(model)

    bins = [0, 0, 0, 0]
    bin_names = ['left', 'right', 'up', 'down']
    for i in range(100):
        action = model.predict(obs, deterministic=False)[0]
        bins[action] += 1

    bin_prob = [b / sum(bins) for b in bins]
    for direction, count, prob in zip(bin_names, bins, bin_prob):
        print(f'{direction}:\n\tSamples: {count}\n\tProbability: {prob:.4%}')

    print()

    for direction, count, prob in zip(bin_names, bins, bin_prob):
        print(f'- {direction}:\t{prob:.4%}')


if __name__ == '__main__':
    input_str = """
        -1 0 4 -1 -1
        -1 0 0 -1 4
        -1 0 0 4 0
        -1 -1 -1 -1 -1
        -1 -1 -1 -1 -1
        """
    test_prediction_for_board(input_str)
