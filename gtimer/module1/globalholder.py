# Global data structures here.

# Yeah make a data structure here.
from focusedstack import FocusedStack
from timerclass import Timer, Loop

timer_stack = FocusedStack(Timer)
loop_stack = FocusedStack(Loop)


#
# Shortcuts
#

# global tf
tf = None  # timer_stack.focus: 'Timer in Focus'

# global lf
lf = None  # loop_stack.focus: 'Loop in Focus'


# TO DO: Automate the making of these shortcuts.
# def focus_shortcut_builder(focus_var, append_name, heap, method, *args, **kwargs):
#     def shortcut(*args, **kwargs):
#         heap.method(*args, **kwargs)
#         global focus_var
#         focus_var = heap.focus
#     shortcut.__name__ = method.__name__ + append_name
#     return shortcut


def create_next_timer(name):
    timer_stack.create_next(name)
    global tf
    tf = timer_stack.focus


def remove_last_timer():
    timer_stack.remove_last()
    global tf
    tf = timer_stack.focus


def focus_backward_timer():
    timer_stack.focus_backward()
    global tf
    tf = timer_stack.focus


def focus_forward_timer():
    timer_stack.focus_forward()
    global tf
    tf = timer_stack.focus


def focus_last_timer():
    timer_stack.focus_last()
    global tf
    tf = timer_stack.focus


def focus_root_timer():
    timer_stack.focus_root()
    global tf
    tf = timer_stack.focus


def create_next_loop(name):
    loop_stack.create_next(name)
    global lf
    lf = loop_stack.focus


def remove_last_loop():
    loop_stack.remove_last()
    global lf
    lf = loop_stack.focus


def focus_backward_loop():
    loop_stack.focus_backward()
    global lf
    lf = loop_stack.focus


def focus_forward_loop():
    loop_stack.focus_forward()
    global lf
    lf = loop_stack.focus


def focus_last_loop():
    loop_stack.focus_last()
    global lf
    lf = loop_stack.focus


def focus_root_loop():
    loop_stack.focus_root()
    global lf
    lf = loop_stack.focus
