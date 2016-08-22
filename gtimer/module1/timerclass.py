# This one holds the Timer class (which will just be a few attributes).
from timeit import default_timer as timer
import timesclass


# Starting simple.
class Timer(object):

    def __init__(self, name=None):
        self.name = name
        self.times = timesclass.Times(name)
        self.in_loop = False
        self.start_t = timer()
        self.last_t = self.start_t


class Loop(object):

    def __init__(self, name=None):
        self.name = name
        self.stamps = list()
        self.itr_stamp_used = dict()
        self.while_condition = True
        # self.start_t


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
