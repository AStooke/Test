

class Input(object):

    def __init__(self, 
                 code, 
                 name=None, 
                 shmem_array=None, 
                 max_shape=None, 
                 typecode=None, 
                 shmem_tag=None,
                 ):
        self.name = name
        self.data = shmem_array
        self.shape = max_shape
        self.typecode = typecode
        self.tag = shmem_tag
        self.code = code


    def assign_indeces(self, n_gpu):
        batch_size = self.max_shape[0]
        assert batch_size >= n_gpu
        worker_size = -(-batch_size // n_gpu)  # (ceiling division)
        boundaries = [worker_size * i for i in range(n_gpu + 1)]
        boundaries[-1] = batch_size
        self.worker_indeces = tuple(
            (boundaries[i], boundaries[i + 1]) for i in range(n_gpu))
