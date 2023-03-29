from typing import Union
import random


class ProbabilityRange:
    """Represents a range of probability values.

    A lower and upper bound are defined, and can be uniformly sampled from."""
    def __init__(self, lower_bound: float, upper_bound: float):
        assert (
                0 <= lower_bound <= 1 and 0 <= upper_bound <= 1
        ), "Probability bounds must be between 0 and 1, inclusive."
        assert (
                lower_bound <= upper_bound
        ), "The lower probability bound must be less than or equal to the upper bound."
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def sample(self) -> float:
        """
        Sample a random probability value within the stored lower and upper bounds.

        :return: A random probability value within the bounds. (float)
        """
        return random.uniform(self.lower_bound, self.upper_bound)
