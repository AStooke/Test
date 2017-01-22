"""
some theano speed tests
"""

import numpy as np
import theano
import theano.tensor as T 
from theano import pp

x = T.dscalar('x')
y = x ** 2
gy = T.grad(y, x)
pp(gy)
f = theano.function([x], gy)
f(4)
