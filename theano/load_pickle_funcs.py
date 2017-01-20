import theano
import pickle
import numpy as np
import gtimer as gt

SIZE = 150
# Create a somewhat complicated theano graph.

# x = theano.shared(np.ones([SIZE, SIZE], dtype='float32'), name='x')
# y = theano.tensor.matrix('y')
# v = theano.tensor.vector('v')
# u = theano.tensor.vector('u')

# z = x.dot(y)
# z1 = z.dot(v)
# z2 = y.dot(u)
# z3 = z1 + z2
# z4 = x.dot(z3)
# z5 = z4.tanh()
# z6 = z5.sum()

# g = theano.grad(z6, wrt=x)

gt.start()

# f = theano.function([y, v, u], [z4, g])

# gt.stamp("f", qp=True)

# with open("f_pre_comp.pkl", "wb") as fil:
# with open("f_post_comp.pkl", "rb") as fil:
# with open("f_pre_comp_new_bknd.pkl", "rb") as fil:
with open("f_post_comp_new_bknd.pkl", "rb") as fil:
    f = pickle.load(fil)

gt.stamp("load_pkl", qp=True)


# theano.printing.debugprint(f)

y_dat = np.ones([SIZE, SIZE], dtype='float32')
v_dat = np.ones([SIZE], dtype='float32')
u_dat = 2 * np.ones([SIZE], dtype='float32')

gt.blank_stamp()

res = f(y_dat, v_dat, u_dat)

gt.stamp("compile", qp=True)

for _ in range(1000):
    res = f(y_dat, v_dat, u_dat)

gt.stamp("run", qp=True)

print(res[0])

# with open("f_post_comp.pkl", "wb") as fil:
#     pickle.dump(f, fil)
