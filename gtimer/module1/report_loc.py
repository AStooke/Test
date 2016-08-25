
"""
Reporting functions acting on locally provided variables (hidden from user).
"""

# A few header formats.
HDR_WIDTH = 12
HDR_PREC = 5
FMT_NAME = "\n{{:<{}}}\t".format(HDR_WIDTH)
FMT_FLT = FMT_NAME + "{{:.{}g}}".format(HDR_PREC)
FMT_GEN = FMT_NAME + "{}"
FMT_INT = FMT_NAME + "{:,}"
# Later, make it so the user can set width, prec with a function call.

#
# Functions to expose elsewhere in package.
#


def write_report(times, include_itrs=True, include_diagnostics=True):
        # if not self._stopped:
        #     raise RuntimeError("Cannot report an active Times structure, must be stopped.")
        rep = "\n---Begin Timer Report---"
        rep += FMT_GEN.format('Timer:', repr(times.name))
        rep += FMT_FLT.format('Total:', times.total)
        rep += FMT_FLT.format('Self:', times.self_t)
        # rep += FMT_FLT.format('Stamps Sum:', times.stamps_sum)
        # if include_diagnostics:
        #     rep += FMT_FLT.format('Self:', times.self)
        #     rep += FMT_FLT.format('Self Agg.:', times.self_agg)
        #     rep += FMT_INT.format('Calls:', times.calls)
        #     rep += FMT_INT.format('Calls Agg.:', times.calls_agg)
        rep += "\n\nIntervals\n---------"
        rep += _report_stamps(times)
        if include_itrs:
            rep_itrs = ''
            rep_itrs += _report_itrs(times)
            if rep_itrs:
                rep += "\n\nLoop Iterations\n---------------"
                rep += rep_itrs
        rep += "\n---End Timer Report---\n"
        return rep


def print_report(include_diagnostics=True):
    rep = write_report(include_diagnostics=include_diagnostics)
    print rep
    return rep


def write_structure(times):
    strct = '\n---Times Data Tree---\n'
    strct += _write_structure(times)
    strct += "\n\n"
    return strct


def _write_structure(times, indent=0):
    strct = "\n{}{}".format(' ' * indent, repr(times.name))
    if times.pos_in_parent:
        strct += " ({})".format(repr(times.pos_in_parent))
    for k, child_list in times.children.iteritems():
        for child in child_list:
            strct += _write_structure(child, indent=indent + 4)
    return strct


def print_structure(times):
    strct = write_structure(times)
    print strct
    return strct


#
# Private helper functions.
#


def _report_stamps(times, indent=0, prec=4):
    rep_stmps = ''
    fmt = "\n{}{{:.<24}} {{:.{}g}}".format(' ' * indent, prec)
    for stamp in times.stamps:  # need to make this ordered again
        rep_stmps += fmt.format("{} ".format(stamp), times.stamps[stamp])
        if stamp in times.children:
            for child in times.children[stamp]:
                rep_stmps += _report_stamps(child, indent=indent + 2)
    return rep_stmps


def _report_itrs(times):
    rep_itrs = ''
    if times.stamps_itrs:
        if times.name:
            rep_itrs += FMT_GEN.format('Timer:', repr(times.name))
        if times.parent is not None:
            rep_itrs += FMT_GEN.format('Parent Timer:', repr(times.parent.name))
            lin_str = _fmt_lineage(_get_lineage(times))
            rep_itrs += FMT_GEN.format('Stamp Lineage:', lin_str)
        rep_itrs += "\n\nIter."
        stamps_itrs_order = []
        is_key_active = []
        for k in times.stamps:  # Need to make this ordered again.
            if k in times.stamps_itrs:
                stamps_itrs_order += [k]
                is_key_active += [True]
        for k in stamps_itrs_order:
            rep_itrs += "\t{:<12}".format(k)  # Stop using tabs
        rep_itrs += "\n-----"
        for k in stamps_itrs_order:
            rep_itrs += "\t------\t"
        itr = 0
        while any(is_key_active):
            next_line = '\n{:<5,}'.format(itr)
            for i, k in enumerate(stamps_itrs_order):
                if is_key_active[i]:
                    try:
                        val = times.stamps_itrs[k][itr]
                        if val < 0.001:
                            prec = 2
                        else:
                            prec = 3
                        next_line += "\t{{:.{}g}}\t".format(prec).format(val)
                    except IndexError:
                        next_line += "\t\t"
                        is_key_active[i] = False
                else:
                    next_line += "\t\t"
            if any(is_key_active):
                rep_itrs += next_line
            itr += 1
        rep_itrs += "\n"
    for _, children in times.children.iteritems():
        for child in children:
            rep_itrs += _report_itrs(child)
    return rep_itrs


def _get_lineage(times):
    if times.pos_in_parent is not None:
        return _get_lineage(times.parent) + (repr(times.pos_in_parent), )
    else:
        return tuple()


def _fmt_lineage(lineage):
    lin_str = ''
    for link in lineage:
        lin_str += "({})-->".format(link)
    try:
        return lin_str[:-3]
    except IndexError:
        pass
