# import os

# os.environ['GTIMER_DISABLE'] = '1'

import testmod

# testmod.submod1.func()
# testmod.func()
# testmod.initialize()
# print testmod


# print dir(testmod)

# reload(testmod)

testmod.submod1.func1()
testmod.submod2.func2()
testmod.submod1.func1()
testmod.submod2.func2()
testmod.submod1.func1()