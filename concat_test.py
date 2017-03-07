import numpy as np
import gtimer as gt

NUM_PATHS = 100
MAX_PATH = 100
OBS_SIZE = 100

# make random data and empty container
data_list = list()
for _ in range(NUM_PATHS):
    path_size = np.random.randint(1, MAX_PATH)
    path = np.random.randn(path_size, OBS_SIZE)
    data_list.append(path)
empty_container = np.empty([int(MAX_PATH * NUM_PATHS * 0.6), OBS_SIZE])
gt.stamp('made', qp=True)

# concatenate into new array
x = np.concatenate(data_list, axis=0)

gt.stamp('concat', qp=True)

start_idx = 0
for path in data_list:
    path_size = path.shape[0]
    empty_container[start_idx:start_idx + path_size] = path
    start_idx += path_size

gt.stamp('write-in', qp=True)


# RESULT: THEY ARE THE SAME SPEED!
