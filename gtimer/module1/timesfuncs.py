# This one holds functions relating the times structures.


def dump_times(times):
    assert not times.children_awaiting, "Oops, all awaiting children should have been put in 'Unassigned' position."
    for k, v in times.children.iteritems():
        for child in v:
            dump_times(child)  # Need to double check this recursion is right.
    rcvr = times.where_to_dump
    if rcvr is not None:
        _merge_times(rcvr, times)
        # Clearing is probably not needed, but at least for now it will prevent redundant dumping.
        # clear_times(times)


# def clear_times(times):
#     times.total = 0.
#     times.stamps.clear()
#     times.stamps_itrs.clear()
#     times.parent = None
#     times.pos_in_parent = None
#     times.children.clear()
#     times.children_awaiting.clear()
#     times.where_to_dump = None


def assign_children(times, name):
    for child_times in times.children_awaiting:
        child_times.pos_in_parent = name
    times.children[name] = times.children_awaiting  # Transfer the list.
    times.children_awaiting = dict()


def l_assign_children(times, name):
    for child_times in times.children_awaiting:
        merged = False
        if name in times.children:
            for old_child in times.children[name]:
                if old_child.name == child_times.name:
                    break
            else:
                old_child = None
            if old_child is not None:
                _merge_times(old_child, child_times)
                merged = True
        if not merged:
            child_times.pos_in_parent = name
            times.children[name] += child_times
    times.children_awaiting = dict()


def _merge_times(rcvr, new):
    rcvr.total += new.total
    _merge_dict(rcvr, new, 'stamps')
    _merge_dict(rcvr, new, 'stamps_itrs')
    for k, v in new.stamps.iteritems():
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


def _merge_dict(rcvr, new, attr):
    rcvr_dict = getattr(rcvr, attr)
    new_dict = getattr(new, attr)
    for k, v in new_dict.iteritems():
        if k in new_dict:
            rcvr_dict[k] += v
        else:
            rcvr_dict[k] = v
