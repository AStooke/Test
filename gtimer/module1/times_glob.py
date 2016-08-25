
"""
Times functions acting on global variables (hidden from user).
"""

import data_glob as g
import times_loc


#
# Functions to expose elsewhere in the package.
#


def assign_children(position):
    for _, child_times in g.rf.children_awaiting.iteritems():
        child_times.pos_in_parent = position
        if position in g.rf.children:
            g.rf.children[position] += [child_times]
        else:
            g.rf.children[position] = [child_times]
    g.rf.children_awaiting.clear()


def l_assign_children(position):
    for _, child_times in g.rf.children_awaiting.iteritems():
        is_prev_child = False
        if position in g.rf.children:
            for old_child in g.rf.children[position]:
                if old_child.name == child_times.name:
                    is_prev_child = True
                    break
            if is_prev_child:
                times_loc.merge_times(old_child, child_times)
        else:
            g.rf.children[position] = []
        if not is_prev_child:
            child_times.pos_in_parent = position
            g.rf.children[position] += [child_times]
    g.rf.children_awaiting.clear()


def dump_times():
    if g.rf.dump is not None:
        times_loc.merge_times(g.rf.dump, g.rf)
