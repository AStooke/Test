
from operator import attrgetter
# def OneAndOnly(inputs):
#     print inputs

class OneAndOnly(object):

    def __init__(self):
        self._yeah = 1

    yeah = property(attrgetter("_yeah"))




