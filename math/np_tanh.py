import gtimer as gt


import numpy as np
gt.stamp('import')


num = 100000
x = np.random.rand(num)
# y = np.zeros(num)
gt.stamp('rand')
# for i, val in enumerate(x):
#     y[i] = np.tanh(val)

z = np.tanh(x)
gt.stamp('tanh')

gt.stop()
print(gt.report())
