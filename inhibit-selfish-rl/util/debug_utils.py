import numpy as np


def numpy_array_info(array, name=None):
    """Prints information about a numpy array."""
    if name is not None:
        print(f"{name}:")
    print("dtype:", array.dtype)
    print("shape:", array.shape)
    print("min:", np.min(array))
    print("max:", np.max(array))
    # print("mean:", np.mean(array))
    # print("std:", np.std(array))
    # print("sum:", np.sum(array))
    # print("unique:", np.unique(array))
    print()
