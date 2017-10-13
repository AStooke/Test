
import numpy as np
import theano
import theano.tensor as T
import time

S = 1000
EPOCHS = 200

x = T.matrix('x')
y = T.matrix('y')

z = T.mean(x.dot(y))

f = theano.function([x, y], z)

x_np = np.random.randn(S, S).astype("float32")
y_np = np.random.randn(S, S).astype("float32")

for _ in range(5):
    r = f(x_np, y_np)

t0 = time.time()
for _ in range(EPOCHS):
    r = f(x_np, y_np)
t1 = time.time()

print("completed {} calls on size {} in {:.3f} s".format(EPOCHS, S, t1 - t0))


