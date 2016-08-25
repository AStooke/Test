
"""
Times functions acting on locally provided variables (hidden from user).
"""

#
# Function to expose elsewhere in the package.
#


def merge_times(rcvr, new, stamps_as_itr=True):
    rcvr.total += new.total
    if stamps_as_itr:
        _merge_stamps_as_itr(rcvr, new)
    _merge_dict(rcvr, new, 'stamps')
    _merge_dict(rcvr, new, 'stamps_itrs')
    _merge_children(rcvr, new)


#
# Private, helper functions.
#


def _merge_children(rcvr, new):
    for child_pos, new_children in new.children.iteritems():
        if child_pos in rcvr.children:
            for new_child in new_children:
                for rcvr_child in rcvr.children[child_pos]:
                    if rcvr_child.name == new_child.name:
                        merge_times(rcvr_child, new_child)
                        # merge_children(rcvr_child, new_child)
                        break
                else:
                    new_child.parent = rcvr
                    rcvr.children[child_pos] += [new_child]
        else:
            for child in new_children:
                child.parent = rcvr
            rcvr.children[child_pos] = new_children
    # Clean up references to old data as we go (not sure if helpful?).
    new.children.clear()


def _merge_dict(rcvr, new, attr):
    rcvr_dict = getattr(rcvr, attr)
    new_dict = getattr(new, attr)
    for k, v in new_dict.iteritems():
        if k in rcvr_dict:
            rcvr_dict[k] += v
        else:
            rcvr_dict[k] = v


def _merge_stamps_as_itr(rcvr, new):
    for k, v in new.stamps.iteritems():
        if k not in new.stamps_itrs:
            if k in rcvr.stamps_itrs:
                rcvr.stamps_itrs[k].append(v)
            else:
                if k in rcvr.stamps:
                    rcvr.stamps_itrs[k] = [rcvr.stamps[k], v]
                else:
                    rcvr.stamps_itrs[k] = [v]
    for k in rcvr.stamps:
        if k not in new.stamps:
            if k in rcvr.stamps_itrs:
                rcvr.stamps_itrs[k].append(0.)
            else:
                rcvr.stamps_itrs[k] = [rcvr.stamps[k], 0.]


# Man, I had just gotten around this incessant merging by having statically
# defined timers, who would accumulate their own data and could be connected
# by linking without having to move any data recursively. Now, with dynamic
# timers, back to having to merge trees (move data recursively) on the fly.
