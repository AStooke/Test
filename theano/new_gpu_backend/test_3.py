import numpy
import theano
from timeit import default_timer as timer
import time

#######################################################
# WITH SHARED VARIABLES

# v01 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
#                     target='dev0')
# v02 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
#                     target='dev0')
# v11 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
#                     target='dev1')
# v12 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
#                     target='dev1')

# v11.transfer('dev1')
# v12.transfer('dev1')
# time.sleep(0.5)
# f = theano.function([], [theano.tensor.dot(v01, v02).transfer('dev0'),
#                          theano.tensor.dot(v11, v12).transfer('dev1')])
# time.sleep(4)
# print("about to make variables on dev0")
# time.sleep(1)
# v1 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'), target='dev0')
# v2 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'), target='dev0')
# print("made them")
# time.sleep(5)
# print("about to transfer to dev1")
# time.sleep(5)
# v1.transfer('dev1')
# v2.transfer('dev1')
# print("transfered")
# time.sleep(5)
# print(dir(v1))

# f = theano.function([], [theano.tensor.dot(v1, v2).transfer('dev1')])

##############################################################

# With regular tensors.
v1_var = theano.tensor.matrix(name='v1', dtype=theano.config.floatX)
v2_var = theano.tensor.matrix(name='v2', dtype=theano.config.floatX)
v3_var = theano.tensor.matrix(name='v3', dtype=theano.config.floatX)



v1_0 = v1_var.transfer('dev0')
v2_0 = v2_var.transfer('dev0')
v1_1 = v1_var.transfer('dev1')
v2_1 = v2_var.transfer('dev1')

# dev = theano.tensor.scalar(name='devic', value=0)

z_var = theano.tensor.dot(v1_var, v2_var)
# f = theano.function([v1_var, v2_var], z_var)
v2_dat0 = numpy.random.random((1024, 1024)).astype('float32')

f2 = theano.function(inputs=[v1_var], outputs=z_var, givens={v2_var: v2_dat0})




# print("\nv1_tran is v1_var: ", v1_tran is v1_var)
# print("\n type(v1_var): ", type(v1_var))
# print("\n type(v1_tran): ", type(v1_tran))
# print(dir(v1_tran))

# u_0 = theano.tensor.dot(v1_0, v2_0)
# u_1 = theano.tensor.dot(v1_1, v2_1)

z0_var = theano.tensor.dot(v1_0, v2_0).transfer('dev0')
z1_var = theano.tensor.dot(v1_1, v2_1).transfer('dev1')

f_0 = theano.function([v1_0, v2_0], z0_var.transfer('dev0'))
f_1 = theano.function([v1_1, v2_1], z1_var.transfer('dev1'))
# f = theano.function([v1_0, v2_0, v1_1, v2_1], [z0_var, z1_var])

v1_dat0 = numpy.random.random((1024, 1024)).astype('float32')
v1_dat1 = numpy.random.random((1024, 1024)).astype('float32')
v2_dat1 = numpy.random.random((1024, 1024)).astype('float32')

# v1_shar = theano.shared(numpy.zeros([1024, 1024], dtype='float32'), target='dev0')
# v2_shar = theano.shared(numpy.zeros([1024, 1024], dtype='float32'), target='dev0')

r_0 = f_0(v1_dat0, v2_dat0)
r_1 = f_1(v1_dat1, v2_dat1)
# r = f(v1_dat0, v2_dat0, v1_dat1, v2_dat1)
r2 = f2(v1_dat0, v2_dat0)
print()
theano.printing.debugprint(f2)
print()

# v1_shar.set_value(v1_dat)
# v2_shar.set_value(v2_dat)

time.sleep(0.5)
t0 = timer()
for i in range(1000):
    r_0 = f_0(v1_dat0, v2_dat0)
    r_1 = f_1(v1_dat1, v2_dat1)
    # nr_0 = numpy.asarray(r_0)
    # nr_1 = numpy.asarray(r_1)
    # r = f(v1_dat0, v2_dat0, v1_dat1, v2_dat1)
t1 = timer()
# print(f.maker.fgraph.toposort())
# print("r[0] type: ", type(r[0]))
# nr = numpy.asarray(r[0])
# print("nr[0] type: ", type(nr))
# time.sleep(0.5)
t1a = timer()
# nr0 = numpy.asarray(r[0])
print(r_0)
# nr0 = numpy.asarray(r_0)
t2 = timer()
# nr1 = numpy.asarray(r[1])
t3 = timer()
print()
# theano.printing.debugprint(f)
# theano.printing.debugprint(f_0)
print()
# theano.printing.debugprint(f_1)
print("\ndot time: ", t1 - t0)
print("nr0 time: ", t2 - t1a)
# print("nr1 time: ", t3 - t2)
print("tots time: ", t3 - t0)
