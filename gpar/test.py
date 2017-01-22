
import os
import numpy as np

import master

y_shar = master.input('y', [5, 3])


master.fork(inputs=None, n_gpu=2)


# os.environ["THEANO_FLAGS"] = "device=cuda0"
import theano

x = theano.shared(np.ones([4, 4], dtype='float32'), 'x')
y = theano.tensor.matrix(name='y', dtype='float32')

# f = theano.function([], x ** 2)
f = master.function(inputs=[y], outputs=theano.tensor.sum(y ** 2, axis=0))

# r1 = f._theano_function()
# print(r1)


# x.set_value(2 * np.ones([4, 4], dtype='float32'))
master.distribute_functions()
# theano.printing.debugprint(f._theano_function)

y_dat = np.ones([5, 3], dtype='float32')
y_dat[0] = 2
y_dat[3] = 3
r = f(y_dat)

print("\nmaster says: \n", r)




















master.close()
