
"""
Reporting functions acting on global variables (all are exposed to user).
"""

import data_glob as g
import report_loc

#
# Reporting functions to expose to the user.
#


def write_report():
    # Write report of the current one:
    return report_loc.write_report(g.rf)


def print_report():
    rep = report_loc.write_report(g.rf)
    print rep
    return rep


def write_structure():
    return report_loc.write_structure(g.rf)


def print_structure():
    strct = report_loc.write_structure(g.rf)
    print strct
    return strct
