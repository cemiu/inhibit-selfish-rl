"""
Author: cemiu (Philipp B.)
This file is used to replace the gym module with gymnasium.
Import this file before importing gym or stable_baselines3.
"""

import sys
import gymnasium

sys.modules["gym"] = gymnasium
