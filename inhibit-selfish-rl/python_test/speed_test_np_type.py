import time

import numpy as np

if __name__ == '__main__':
    temp = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)

    start_time = time.time()

    for i in range(1000000):
        a = np.zeros((15, 15), dtype=np.int8)
        b = np.concatenate((a.flatten(), temp))
        # print(b_training_logs)
        # numpy_array_info(b_training_logs)

    end_time = time.time()
    print('time: ', end_time - start_time)

# 17.516998767852783 sec int8 -> int64
# 17.14709711074829 sec int64 -> int64
