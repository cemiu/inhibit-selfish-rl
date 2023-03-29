from .models import *

# # Get the current namespace
# namespace = globals()
#
# # Filter variables that do not start with '_' and export them
# __all__ = [name for name in namespace if not name.startswith('_')]


__all__ = [
    'B_DOOR_MUTEX',
    'B_DOOR_SHARED',
    'B_SLOWDOWN_MUTEX',
    'B_SPEEDUP_MUTEX',

    'REWARD_NET_DOOR_MUTEX',
    'REWARD_NET_DOOR_SHARED',
    'REWARD_NET_SLOWDOWN_MUTEX',
    'REWARD_NET_SPEEDUP_MUTEX',

    'TEST_DOOR_MUTEX_10x10_A_SELFISH',
    'TEST_DOOR_MUTEX_10x10_A_DOOR_BLOCK_REWARD',
    'TEST_DOOR_MUTEX_10x10_A_B_REWARD_0_25',
    'TEST_DOOR_MUTEX_10x10_A_B_REWARD_1',
    'TEST_DOOR_MUTEX_10x10_A_B_REWARD_5',
    'TEST_DOOR_MUTEX_10x10_B_150K',
    'TEST_DOOR_MUTEX_10x10_B_1M',
    'TEST_DOOR_MUTEX_10x10_B_5M',
    'TEST_DOOR_MUTEX_10x10_B_REWARD_LEARNED',
    'TEST_DOOR_MUTEX_10x10_REWARD_MODEL',
    'TEST_DOOR_SHARED_10x10_B_1M',
    'TEST_DOOR_SHARED_10x10_B_10M',
    'TEST_DOOR_SHARED_10x10_REWARD_MODEL'
]
