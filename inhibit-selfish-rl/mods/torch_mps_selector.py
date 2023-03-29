from collections.abc import Callable
from functools import partial


import torch as th
from stable_baselines3.common import utils

"""This modifies the PyTorch device selection, to allow for MPS (Metal Performance shaders) devices."""

_original_get_device = None


def patch_mps_device_selection(modified: bool = True):
    """Patches the get_device method to select for MPS device."""
    if modified:
        _patch_inject_mps_device()
    else:
        _unpatch_inject_mps_device()


def _inject_mps_device(
        source_function: Callable[[str | th.device], th.device],
        device: str | th.device = "auto",
) -> th.device:
    """Injects the MPS device into the PyTorch device selection.

    The MPS device is only available if the MPS backend is available.

    If the MPS backend is not available, this function will return the original device.
    """
    is_requesting_cpu = device == "cpu"
    device = source_function(device)  # use the default selector, picking CUDA or cpu

    # If the device is a CPU, and a CPU was not explicitly requested, try to use MPS
    if device.type == th.device("cpu").type and not is_requesting_cpu:
        try:
            # noinspection PyUnresolvedReferences
            if th.backends.mps.is_available():
                device = th.device("mps")
        except AttributeError:  # MPS supporting version of PyTorch not installed
            pass

    return device


def _patch_inject_mps_device():
    global _original_get_device
    if _original_get_device is None:
        _original_get_device = utils.get_device
    utils.get_device = partial(_inject_mps_device, source_function=_original_get_device)


def _unpatch_inject_mps_device():
    if _original_get_device is not None:
        utils.get_device = _original_get_device
