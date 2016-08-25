
"""
Timer data and state containers (hidden from user).
"""

from timeit import default_timer as timer


class Timer(object):
    """ Primarily contains status values."""
    def __init__(self, name=None, loop_depth=0, **kwargs):
        self.name = name
        self.times = Times(name, **kwargs)
        self.loop_depth = loop_depth
        self.stopped = False
        self.paused = False
        self.self_t = 0.
        self.start_t = timer()
        self.last_t = self.start_t


class Times(object):
    """ Primarily contains data values.

    These might be exposed to user later, if they want to explore
    the structure...maybe should write-protect.
    """

    def __init__(self, name=None, parent=None, pos_in_parent=None):
        self.name = name
        self.total = 0.
        self.self_t = 0.
        self.stamps = dict()
        self.stamps_itrs = dict()
        self.stamps_sum = 0.
        self.parent = parent  # refer to another Times instance.
        self.pos_in_parent = pos_in_parent  # refer to a stamp name.
        self.children = dict()  # key: position in self, value: list of Times instances.
        self.children_awaiting = dict()  # key: name of child, value: a Times instance.
        self.dump = None  # refer to another Times instance.


# class Loop(object):

#     def __init__(self, save_itrs=True):
#         self.name = None
#         self.save_itrs = save_itrs
#         self.reg_stamps = list()
#         self.stamp_used = dict()
#         self.start_t = timer()
#         self.while_condition = True


# class TmpData(object):

#     def __init__(self):
#         self.self_t = 0.
#         self.calls = 0.
#         self.times = timesclass.Times()


# class Timer(object):

#     def __init__(self, name, save_self_itrs=True, save_loop_itrs=True):
#         self.name = name
#         self.save_itrs = save_self_itrs
#         self.save_loop_itrs = save_loop_itrs
#         self.reg_stamps = list()
#         self.in_loop = False
#         self.active = True
#         self.paused = False
#         self.start_t = timer()
#         self.last_t = self.start_t
#         self.tmp = TmpData()
#         self.loop = None
