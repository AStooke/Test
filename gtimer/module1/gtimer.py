# Main file that gets imported.

# Conditional importing of subfunctions, given environment
# variables, goes here.

# Functions to expose to the user.
from timer_glob import *
from loop import *
from timer_mgmt import *
from report_glob import *




#
#                  ...TO DO...
#
# 7. Once the Times class is stabilized, all the reporting
#     stuff that I wanted to do before.
# 8. How to handle multiple separate heap (contexts)?
# 11. make my own error classes?
# 15. DUH: MULTIPROCESSING!!!
# 16. Reporting in the middle of timing...?
# 17. Test behavior in un-timed loop inside of timed loop
# 18. Make a reporting function that takes many timers (i.e. many
#     different runs of the same program), and makes tables of all
#     their stamps.  So that each table has TIMER on one axis and
#     STAMP on the other...yes this is what I'm really after!
# .
# .
# N+1. Automate the shortcut building.
#
#
#               ... LOW PRIORITY...
# 14. NO_CHECK mode. (well let's see what the self times are)
# 13. DISABLE mode.
# 10. stamp itr statistics (running avg, running stdev?)
#
#
#               .. to DONE...
# 5. Self time and all that. (aggregation)
# 9. Stamps ordered AND register stamps
# 18. loop break / loop continue
# 6. Allow duplicate.
# 12. pause & resume
# 3. Auto-focus manager for subfunctions.
# 4. Test, test!
# 1. **Get the child-parent timer relationships working.**
# 2. Test, test
# 9. Get code files organized into global funcs and private funcs.
# N. Think about how much of focus manager to expose
#     to the user for manual manipulation. A: timer_mgmt
#