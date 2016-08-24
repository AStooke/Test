# Functions for creating new timers, closing old ones,
# and handling the relationships of the timers.
#

import data_glob as g
import timer_glob


def open_new_timer(name, in_loop=False):
    if name == 'root':
        raise ValueError('Timer name "root" is reserved.')
    if name in g.rf.children_awaiting:
        dump = g.rf.children_awaiting[name]
        g.create_next_timer(name, in_loop=in_loop)
        g.rf.dump = dump
    else:
        g.create_next_timer(name, times_parent=g.rf, in_loop=in_loop)
        new_times = g.rf
        g.focus_backward_timer()
        g.rf.children_awaiting[name] = new_times
        g.focus_forward_timer()


def close_last_timer():
    if g.tf.name == 'root':
        raise RuntimeError('Attempted to close root timer, can only stop it.')
    if not g.tf.stopped:
        timer_glob.stop()
    g.remove_last_timer()


def timer_wrap(func):
    def timer_wrapped(*args, **kwargs):
        open_new_timer(name=func.__name__)
        result = func(*args, **kwargs)
        close_last_timer()
        return result
    return timer_wrapped
