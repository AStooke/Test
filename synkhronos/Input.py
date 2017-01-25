
from util import NP_TO_C_TYPE


class Input(object):

    def __init__(self,
                 code,
                 dtype,
                 name=None,
                 shape=None,
                 shmem_array=None,
                 ):
        self.code = code
        self.ctype = NP_TO_C_TYPE.get(dtype, None)
        if self.ctype is None:
            raise TypeError("Numpy/Theano type: ", dtype, " not supported.")
        self.name = name
        self.np_array = shmem_array
        self.shmem_tag = code


    def assign_indeces(self, n_gpu):
        batch_size = self.max_shape[0]
        assert batch_size >= n_gpu
        worker_size = -(-batch_size // n_gpu)  # (ceiling division)
        boundaries = [worker_size * i for i in range(n_gpu + 1)]
        boundaries[-1] = batch_size
        self.worker_indeces = tuple(
            (boundaries[i], boundaries[i + 1]) for i in range(n_gpu))
