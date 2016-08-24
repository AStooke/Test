
"""
All global data (and timer state information) resides here.
"""

from focusedstack import FocusedStack
from timer_classes import Timer, Loop


timer_stack = FocusedStack(Timer)
loop_stack = FocusedStack(Loop)


#
# Shortcut variables and shortcut functions.
#

tf = None  # timer_stack.focus: 'Timer in Focus'
rf = None  # timer_stack.focus.times: 'Times (Record) in Focus'
lf = None  # loop_stack.focus: 'Loop in Focus'


# TO DO: Automate the making of these shortcuts.
#
# def focus_shortcut_builder(focus_var, append_name, heap, method, *args, **kwargs):
#     def shortcut(*args, **kwargs):
#         heap.method(*args, **kwargs)
#         global focus_var
#         focus_var = heap.focus
#     shortcut.__name__ = method.__name__ + append_name
#     return shortcut
#


def create_next_timer(name, **kwargs):
    global tf, rf
    tf = timer_stack.create_next(name, **kwargs)
    rf = tf.times


# Initialize the first member of the timer stack,
# user may not remove this one.
create_next_timer('root')


def remove_last_timer():
    global tf, rf
    tf = timer_stack.remove_last()
    if tf is not None:
        rf = tf.times
    else:
        rf = None


def pop_last_timer():
    global tf, rf
    last_timer = timer_stack.pop_last()
    tf = timer_stack.focus
    if tf is not None:
        rf = tf.times
    else:
        rf = None
    return last_timer


def focus_backward_timer():
    global tf, rf
    tf = timer_stack.focus_backward()
    if tf is not None:
        rf = tf.times
    else:
        rf = None


def focus_forward_timer():
    global tf, rf
    tf = timer_stack.focus_forward()
    if tf is not None:
        rf = tf.times
    else:
        rf = None


def focus_last_timer():
    global tf, rf
    tf = timer_stack.focus_last()
    if tf is not None:
        rf = tf.times
    else:
        rf = None


def focus_root_timer():
    global tf, rf
    tf = timer_stack.focus_root()
    if tf is not None:
        rf = tf.times
    else:
        rf = None


def create_next_loop(name=None):
    global lf
    lf = loop_stack.create_next(name)


def remove_last_loop():
    global lf
    lf = loop_stack.remove_last()


def focus_backward_loop():
    global lf
    lf = loop_stack.focus_backward()


def focus_forward_loop():
    global lf
    lf = loop_stack.focus_forward()


def focus_last_loop():
    global lf
    lf = loop_stack.focus_last()


def focus_root_loop():
    global lf
    lf = loop_stack.focus_root()
