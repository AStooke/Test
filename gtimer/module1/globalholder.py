# Global data structures here.

# Yeah make a data structure here.
from focusedstack import FocusedStack
from timerclass import Timer, Loop, Times

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


def create_next_timer(name):
    global tf, rf
    if tf is not None:
        if name in rf.children_awaiting:
            dump_location = rf.children_awaiting[name]
            first_dump = False
        else:
            dump_location = Times(name=name, parent=rf)
            rf.children_awaiting[name] = dump_location
            first_dump = True
    else:
        first_dump = True
        dump_location = None
    tf = timer_stack.create_next(name=name, first_dump=first_dump)
    rf = tf.times
    rf.dump_location = dump_location


def remove_last_timer():
    global tf, rf
    tf = timer_stack.remove_last()
    if tf is not None:
        rf = tf.times


def focus_backward_timer():
    global tf, rf
    tf = timer_stack.focus_backward()
    if tf is not None:
        rf = tf.times


def focus_forward_timer():
    global tf, rf
    tf = timer_stack.focus_forward()
    if tf is not None:
        rf = tf.times


def focus_last_timer():
    global tf, rf
    tf = timer_stack.focus_last()
    if tf is not None:
        rf = tf.times


def focus_root_timer():
    global tf, rf
    tf = timer_stack.focus_root()
    if tf is not None:
        rf = tf.times


def create_next_loop(name):
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
