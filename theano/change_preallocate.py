
import theano

from theano import config

print("imported preallocate: ", config.gpuarray.preallocate)

# from theano.configparser import change_flags
from change_flags import ChangeFlags


# preallocate_changer = change_flags(args={"gpuarray.preallocate": 0.5})

with preallocate_changer:
    import theano.gpuarray
    theano.gpuarray.use('cuda')

# class GpuLoader(object):
#     @change_flags(args={"gpuarray.preallocate": 0.5})
#     def load_gpu(self):
#         import theano.gpuarray
#         theano.gpuarray.use('cuda')

# load_gpu()
# gpu_loader = GpuLoader()
# gpu_loader.load_gpu()
