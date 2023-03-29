import matplotlib.pyplot as plt
import numpy as np

"""
This script demonstrates the relative movement speed of object B with respect to object A after n interactions between them.
It visualizes the movement speed of B for two different scenarios: one where B slows down relative to A ($f(x) = \frac{1}{(x + 1)^{0.6}}$), 
and another where B speeds up, starting slow ($f(x) = 1 - e^{-0.15(x + 0.5)}$).
"""


def plot_function_values(func, n, title):
    """
    Plot the values of a function for the first n integers (0 to n).

    Parameters:
    func (function): A function that maps an integer to a float.
    n (int): The number of integers to evaluate the function for.
    title (str): The title of the plot.
    """
    x_values = range(n + 1)

    y_values = [func(x) for x in x_values]

    fig = plt.figure(facecolor='white')
    plt.title(title)

    # Create a colourmap based on the height of the bars
    colours = np.array(y_values)
    norm = plt.Normalize(colours.min(), colours.max())
    cmap = plt.cm.Spectral

    plt.bar(x_values, y_values, color=cmap(norm(colours)))

    plt.xlabel('Number of interactions between agents')
    plt.ylabel('Movement speed of B')
    # ax = plt.gca()

    plt.show()


# Define a function that slows down B's movement relative to A
def slowdown_function(x):
    return 1 / (x + 1) ** 0.6


# Define a function that speeds up B's movement relative to A
def speedup_function(x):
    return 1 - np.exp(-0.6*(x + 0.1))


if __name__ == '__main__':
    # Plot the slowdown function
    plot_function_values(slowdown_function, 10, 'Movement speed of B (slowing down: $f(x) = \\frac{1}{(x + 1)^{0.6}}$)')

    # Plot the speedup function
    plot_function_values(speedup_function, 10, 'Movement speed of B (speeding up: $f(x) = 1 - e^{-0.6(x + 0.1)}$)')
