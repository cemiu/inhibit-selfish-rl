from collections.abc import Callable, Iterable
from typing import Any, Optional
import random
import string


def human_num(num):
    """Converts a number to a human-readable string.
    Adapted from: https://stackoverflow.com/a/45846841
    Distributed under: CC BY-SA 3.0"""
    if num == 1:
        return 'operation'
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def round_elements(
        iterable: Iterable,
        round_to: int = 2,
        reduce: Callable[[Iterable], Any] | None = None,
) -> Iterable:
    """Takes any iterable of arbitrary elements and rounds all floating point numbers.

    By default, the return value is a list, but this can be changed by passing a different reduce function.
    If the reduce function is None, the return value is a list.

    Args:
        :param iterable: the iterable to round
        :param round_to: the number of decimal places to round to (default: 2)
        :param reduce: the reduce function to use to reduce the map object (default: list)
    """
    rounded = map(lambda elem: round(elem, round_to) if isinstance(elem, float) else elem, iterable)
    if reduce is None:
        return list(rounded)
    return reduce(rounded)


def convert_case(s: str, drop_count: int = 0) -> str:
    """Converts a string in UpperCamelCase to snake_case format.
    Optionally drops the last n parts of the string. For example, if drop_count is 1, then
    "UpperCamelCase" will be converted to "upper_camel".

    Args:
        s: The string to convert.
        drop_count: The number of parts to drop from the end of the string.

    Returns:
        The string in upper_camel_case format, less the last n parts.
    """
    output = ''
    for i, c in enumerate(s):
        if i != 0 and c.isupper():
            output += '_'
        output += c.lower()

    if drop_count > 0:
        parts = output.split('_')
        output = '_'.join(parts[:-drop_count])
    return output


def generate_random_string(n: int) -> str:
    """Generates a random string of n uppercase letters and numbers.

    Args:
        n: The length of the string to generate.

    Returns:
        A random string of n letters and numbers.
    """
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(n))


class ActionTable:
    """Class for printing table of actions taken by agents.

    Input is a list of 4 numbers, representing the number of times each action was taken.
    Order: left, right, up, down."""
    def __init__(self, immediate=False):
        self.lines = []
        self.immediate = immediate
        self.printed_header = False

    # noinspection PyMethodMayBeStatic
    def _print_header(self):
        print("+------+-------+-------+-------+-------+")
        print("| in % | left  | right | up    | down  |")
        print("+------+-------+-------+-------+-------+")

    # noinspection PyMethodMayBeStatic
    def _print_line(self, line_num, line, extra: Optional[str]):
        total_actions = sum(line)
        percentages = [f"{100 * a / total_actions:.0f}" for a in line]
        print(f"| {line_num:>3}  | {percentages[0]:>5} | {percentages[1]:>5} | "
              f"{percentages[2]:>5} | {percentages[3]:>5} |", end="")
        if extra:
            print(f" {extra}")
        else:
            print()

    def add_line(self, line, extra: Optional[str] = None):
        self.lines.append((line, extra))
        if self.immediate:
            if not self.printed_header:
                self._print_header()
                self.printed_header = True
            self._print_line(len(self.lines), line, extra)

    def print_table(self):
        self._print_header()
        for i, (line, extra) in enumerate(self.lines):
            self._print_line(i + 1, line, extra)
        self.print_footer()

    # noinspection PyMethodMayBeStatic
    def print_footer(self):
        print("+------+-------+-------+-------+-------+")

    def done(self):
        if not self.immediate:
            self.print_table()
        else:
            self.print_footer()


def safe_min(*items):
    """Returns the minimum of the given items, or None if there are no items."""
    current_min = None
    for item in items:
        if current_min is None or (item is not None and item < current_min):
            current_min = item

    return current_min

# if __name__ == '__main__':
#     x = ['testing', 42, 3.14, True, None, [1, 2, 3], 0.13823113439995532, 0.9236244401791427]
#     rounded_list = round_elements(x)
