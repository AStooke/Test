# This one holds functions relating the times structures.

import globalholder as g

#
# Functions acting on global.
#


def assign_children(name):
    for child_times in g.rf.children_awaiting:
        child_times.pos_in_parent = name
    g.rf.children[name] = g.rf.children_awaiting  # Transfer the list.
    g.rf.children_awaiting.clear()


def l_assign_children(name):
    for child_times in g.rf.children_awaiting:
        merged = False
        if name in g.rf.children:
            for old_child in g.rf.children[name]:
                if old_child.name == child_times.name:
                    break
            else:
                old_child = None
            if old_child is not None:
                _merge_times(old_child, child_times)
                merged = True
        if not merged:
            child_times.pos_in_parent = name
            g.rf.children[name] += child_times
    g.rf.children_awaiting.clear()


def dump_times():
    rcvr = g.rf.dump_location
    _merge_times(rcvr, g.rf, stamps_as_itr=(not g.tf.first_dump))
    _merge_children(rcvr, g.rf)


#
# Helper functions acting on local.
#


def _merge_times(rcvr, new, stamps_as_itr=True):
    rcvr.total += new.total
    _merge_dict(rcvr, new, 'stamps')
    _merge_dict(rcvr, new, 'stamps_itrs')
    if stamps_as_itr:
        _merge_stamps_as_itr(rcvr, new)


def _merge_dict(rcvr, new, attr):
    rcvr_dict = getattr(rcvr, attr)
    new_dict = getattr(new, attr)
    for k, v in new_dict.iteritems():
        if k in new_dict:
            rcvr_dict[k] += v
        else:
            rcvr_dict[k] = v


def _merge_stamps_as_itr(rcvr, new):
    for k, v in new.stamps.iteritems():
        if k not in new.stamps_itrs:
            if k in rcvr.stamps_itrs:
                rcvr.stamps_itrs[k].append(v)
            else:
                rcvr.stamps_itrs[k] = [v]
    for k in rcvr.stamps:
        if k not in new.stamps:
            if k in rcvr.stamps_itrs:
                rcvr.stamps_itrs[k].append(0.)
            else:
                rcvr.stamps_itrs[k] = [0.]


def _merge_children(rcvr, new):
    for child_pos, new_children in new.children.iteritems():
        if child_pos in rcvr:
            for new_child in new_children:
                for rcvr_child in rcvr[child_pos]:
                    if rcvr_child.name == new_child.name:
                        _merge_times(rcvr_child, new_child)
                        _merge_children(rcvr_child, new_child)
                else:
                    rcvr.children[child_pos] += [new_child]
        else:
            rcvr.chlidren[child_pos] = new_children
    # Clean up references to old data as we go (not sure if helpful?).
    new.children.clear()


# Man, I had just gotten around this incessant merging by having statically
# defined timers, who would accumulate their own data and could be connected
# by linking without having to move any data recursively. Now, with dynamic
# timers, back to having to merge trees (move data recursively) on the fly.
