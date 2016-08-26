
"""
Reporting functions acting on locally provided variables (hidden from user).
"""

from data_glob import UNASGN


# A few header formats.
HDR_WIDE = 20
HDR_SHRT = 10
HDR_PREC = 4
FMT_BASE = "\n{{:<{}}}"
FMT_NAME = FMT_BASE.format(HDR_WIDE)
FMT_NAME_SHRT = FMT_BASE.format(HDR_SHRT)
FMT_FLT = FMT_NAME + "{{:.{}g}}".format(HDR_PREC)
FMT_GEN = FMT_NAME + "{}"
FMT_INT = FMT_NAME + "{:,}"
FMT_GEN_SHRT = FMT_NAME_SHRT + "{}"
STMP_NM_SZ = 24
STMP_PREC = 4
TAB_WIDTH = 2
ITR_WIDTH = 6
NAME_WIDTH = 14
TAB = ' ' * TAB_WIDTH
ITR_SPC = ' ' * (TAB_WIDTH + NAME_WIDTH - ITR_WIDTH)

# Later, make it so the user can set width, prec with a function call?

DELIM = '\t'
D_HDR = "{{}}{}{{}}\n".format(DELIM)
IDT_SYM = '+'


#
# Functions to expose elsewhere in package.
#

def delim_report(times, include_itrs=True):
    rep = "Timer Report\n"
    rep += D_HDR.format('Timer Name', times.name)
    rep += D_HDR.format('Total Time', times.total)
    rep += D_HDR.format('Self Time Agg', times.self_agg)
    rep += D_HDR.format('Self Time Cut', times.self_cut)
    rep += "\n\nIntervals\n"
    rep += _delim_stamps(times)
    if include_itrs:
        rep_itrs = ''
        rep_itrs += _report_itrs(times)
        if rep_itrs:
            rep += "\n\nLoop Iterations\n"
            rep += rep_itrs
    return rep


def write_report(times, include_itrs=True):
        # if not self._stopped:
        #     raise RuntimeError("Cannot report an active Times structure, must be stopped.")
        rep = "\n---Begin Timer Report ({})---".format(times.name)
        rep += FMT_GEN.format('Timer:', times.name)
        rep += FMT_FLT.format('Total Time (s):', times.total)
        rep += FMT_FLT.format('Self Time Agg:', times.self_agg)
        rep += FMT_FLT.format('Self Time Cut:', times.self_cut)
        rep += "\n\n\nIntervals\n---------"
        rep += _report_stamps(times)
        if include_itrs:
            rep_itrs = ''
            rep_itrs += _report_itrs(times)
            if rep_itrs:
                rep += "\n\n\nLoop Iterations\n---------------"
                rep += rep_itrs
        rep += "\n---End Timer Report ({})---\n".format(times.name)
        return rep


def print_report(include_itrs=True):
    rep = write_report(include_itrs)
    print rep
    return rep


def write_structure(times):
    strct = '\n---Times Data Tree---\n'
    strct += _write_structure(times)
    strct += "\n\n"
    return strct


def _write_structure(times, indent=0):
    strct = "\n{}{}".format(' ' * indent, times.name)
    if times.pos_in_parent:
        strct += " ({})".format(times.pos_in_parent)
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


def _report_stamps(times, indent=0):
    rep_stmps = ''
    fmt = "\n{}{{:.<{}}} {{:.{}g}}".format(' ' * indent, STMP_NM_SZ, STMP_PREC)
    stamps = times.stamps
    for stamp in stamps.order:
        rep_stmps += fmt.format("{} ".format(stamp), stamps.cum[stamp])
        if stamp in times.children:
            for child in times.children[stamp]:
                rep_stmps += _report_stamps(child, indent=indent + 2)
    if UNASGN in times.children:
        rep_stmps += "\n{}{}".format(' ' * indent, UNASGN)
        for child in times.children[UNASGN]:
            rep_stmps += _report_stamps(child, indent=indent + 2)
    return rep_stmps


def _delim_stamps(times, indent=0):
    stamps = times.stamps
    rep_stmps = ''
    for stamp in stamps.order:
        rep_stmps += "{}{}{}{}\n".format(IDT_SYM * indent, stamp, DELIM, stamps.cum[stamp])
        for stamp in times.children:
            for child in times.children[stamp]:
                rep_stmps += _delim_stamps(child, indent=indent + 1)
    if UNASGN in times.children:
        rep_stmps += "\n{}{}".format(IDT_SYM * indent, UNASGN)
        for child in times.children[UNASGN]:
            rep_stmps += _delim_stamps(child, indent=indent + 1)


def _report_itrs(times):
    rep_itrs = ''
    stamps = times.stamps
    if stamps.itrs:
        rep_itrs += FMT_GEN_SHRT.format('Timer:', times.name)
        if times.parent is not None:
            lin_str = _fmt_lineage(_get_lineage(times))
            rep_itrs += FMT_GEN_SHRT.format('Lineage:', lin_str)
        rep_itrs += "\n\nIter."
        itrs_order = []
        is_key_active = []
        for stamp in stamps.order:
            if stamp in stamps.itrs:
                itrs_order += [stamp]
                is_key_active += [True]  # (List needed for any() usage.)
        for stamp in itrs_order:
            rep_itrs += "{0}{{:>{1}.{1}}}".format(TAB, NAME_WIDTH).format(stamp)
        rep_itrs += "\n-----"
        for stamp in itrs_order:
            rep_itrs += "{}{{:>{}}}".format(TAB, NAME_WIDTH).format('------')
        itr = 0
        while any(is_key_active):  # (Must be a list)
            next_line = '\n{:<5,}'.format(itr)
            for i, stamp in enumerate(itrs_order):
                if is_key_active[i]:
                    try:
                        val = stamps.itrs[stamp][itr]
                        next_line += "{}{{:{}.2f}}".format(ITR_SPC, ITR_WIDTH).format(val)
                    except IndexError:
                        next_line += "{}{}".format(TAB, ' ' * NAME_WIDTH)
                        is_key_active[i] = False
                else:
                    next_line += "{}{}".format(TAB, ' ' * NAME_WIDTH)
            if any(is_key_active):
                rep_itrs += next_line
            itr += 1
        rep_itrs += "\n"
    for _, children in times.children.iteritems():
        for child in children:
            rep_itrs += _report_itrs(child)
    return rep_itrs


def _delim_itrs(times):
    rep_itrs = ''
    stamps = times.stamps
    if stamps.itrs:
        rep_itrs += "Timer{}{}\n".format(DELIM, times.name)
        lin_str = _fmt_lineage(_get_lineage(times))
        rep_itrs += "Lineage{}{}\n".format(DELIM, lin_str)
        rep_itrs += "\n\nIter."
        itrs_order = []
        is_key_active = []
        for stamp in stamps.order:
            if stamp in stamps.itrs:
                itrs_order += [stamp]
                is_key_active += [True]
        for stamp in itrs_order:
            rep_itrs += "{}{}".format(DELIM, stamp)
        itr = 0
        while any(is_key_active):
            next_line = "\n{}".format(itr)
            for i, stamp in enumerate(itrs_order):
                if is_key_active[i]:
                    try:
                        val = stamp.itrs[stamp][itr]
                        next_line += "{}{}".format(DELIM, val)
                    except IndexError:
                        next_line += "{}".format(DELIM)
                        is_key_active[i] = False
                else:
                    next_line += "{}".format(DELIM)
            if any(is_key_active):
                rep_itrs += next_line
            itr += 1
        rep_itrs += "\n"
    for _, children in times.children.iteritems():
        for child in children:
            rep_itrs += _delim_itrs(child)
    return rep_itrs


def _get_lineage(times):
    if times.parent is not None:
        return _get_lineage(times.parent) + ((times.parent.name, times.pos_in_parent), )
    else:
        return tuple()


def _fmt_lineage(lineage):
    lin_str = ''
    for link in lineage:
        lin_str += "{} ({})--> ".format(link[0], link[1])
    try:
        return lin_str[:-4]
    except IndexError:
        pass
