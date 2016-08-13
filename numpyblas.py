import numpy
import sys
import timeit

try:
    import numpy.core._dotblas
    print 'FAST BLAS'
except ImportError:
    print 'slow blas'

print "version:", numpy.__version__
print "maxint:", sys.maxint
print

x = numpy.random.random((5000,5000))

setup = "import numpy; x = numpy.random.random((5000,5000))"
count = 5

t = timeit.Timer("numpy.dot(x, x.T)", setup=setup)
print "dot:", t.timeit(count)/count, "sec"