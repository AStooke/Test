
"""
Functions for creating new timers, closing old ones, and handling the
relationships of the timers (all exposed to user).
"""

import data_glob as g
import timer_glob


def open_next_timer(name):
    if name == 'root':
        raise ValueError('Timer name "root" is reserved.')
    if g.entering_named_loop:
        if name in g.rf.children:
            assert len(g.rf.children[name]) == 1  # There should only be one child.
            dump = g.rf.children[name][0]
            g.create_next_timer(name, loop_depth=1)
            g.rf.dump = dump
        else:
            # No previous, write directly to assigned child in parent times.
            g.create_next_timer(name, loop_depth=1, parent=g.rf, pos_in_parent=name)
            new_times = g.rf
            g.focus_backward_timer()
            g.rf.children[name] = [new_times]
            g.focus_forward_timer()
    else:  # (When user calls, this branch always taken)
        if name in g.rf.children_awaiting:
            # Previous dump exists.
            # (e.g. multiple calls of same wrapped function between stamps)
            dump = g.rf.children_awaiting[name]
            g.create_next_timer(name)
            g.rf.dump = dump
        else:
            # No previous, write directly to awaiting child in parent times.
            g.create_next_timer(name, parent=g.rf)
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


def wrap(func):
    def timer_wrapped(*args, **kwargs):
        open_next_timer(name=func.__name__)
        result = func(*args, **kwargs)
        close_last_timer()
        return result
    return timer_wrapped
