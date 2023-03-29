from typing import Tuple

import numpy as np


def gen_array(size):
    """Generates a 2d array of size x size

    The format is as follows:
    0 1 2 3 4 ...
    1 2 3 4 5 ...
    2 3 4 5 6 ...
    3 4 5 6 7 ...
    4 5 6 7 8 ...
    ...
    """
    arr = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            arr[i][j] = i + j

    return arr


def extract_sub_matrix(
        arr: np.ndarray,
        point: tuple[int, int],
        radius: int
) -> np.ndarray:
    """Extracts a sub matrix of size 2*radius+1 from a 2d array.

    Out of bounds values are set to -1.

    Args:
        :param arr: The array to extract the sub matrix from.
        :param point: The point to extract the sub matrix from.
        :param radius: The radius of the sub matrix.
    """
    x, y = point
    size = 2 * radius + 1
    sub_matrix = np.full((size, size), -1, dtype=arr.dtype)
    for i in range(size):
        for j in range(size):
            if 0 <= x + i - radius < arr.shape[0] and 0 <= y + j - radius < arr.shape[1]:
                sub_matrix[i][j] = arr[x + i - radius][y + j - radius]

    return sub_matrix


def is_inside_submatrix(pos: Tuple[int, int], row: int, col: int, radius: int) -> bool:
    """Checks if a position is inside a sub matrix of a given radius."""
    return pos[0] - radius <= row <= pos[0] + radius and pos[1] - radius <= col <= pos[1] + radius


def calculate_statistics(arr: np.ndarray) -> tuple[float, float, float]:
    """
    Calculates the mean, minimum, and maximum values of a 2D array.
    Args:
        arr (np.ndarray): A 2D NumPy array of values.

    Returns:
        tuple: A tuple of three floating-point values representing the mean, minimum,
               and maximum values in the array.
    """
    mean_val = float(np.nanmean(arr))
    min_val = float(np.nanmin(arr))
    max_val = float(np.nanmax(arr))

    return mean_val, min_val, max_val


def normalise_arrays(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Normalise arrays such that the values range from 0 to the maximum value.

    Args:
        *arrays: One or more numpy arrays to be normalised.

    Returns:
        A tuple of normalised numpy arrays.
    """
    # Find the global maximum value across all input arrays
    global_max = np.max([np.nanmax(array) for array in arrays])

    # If global_max is 0, return the input arrays as they are
    if global_max == 0:
        return arrays

    # Normalise each input array by dividing by the global maximum value
    normalised_arrays = tuple(array / global_max for array in arrays)

    return normalised_arrays
