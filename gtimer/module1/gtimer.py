# Main file that gets imported.

# Conditional importing of subfunctions, given environment
# variables, goes here.

# Functions to expose to the user.
from timer_glob import *
from loop import timed_for, timed_while, break_for
from timer_mgmt import *
from report_glob import *




#
#                  ...TO DO...
#
# 5. Self time and all that. (aggregation)
# 7. Once the Times class is stabilized, all the reporting
#     stuff that I wanted to do before.
# 8. How to handle multiple separate heap (contexts)?
# 9. Stamps ordered.
# 10. stamp itr statistics (running avg, running stdev?)
# 11. make my own error classes?
# 13. DISABLE mode.
# 14. NO_CHECK mode. (well let's see what the self times are)
# 15. DUH: MULTIPROCESSING!!!
# 16. Reporting in the middle of timing...?
# 17. Test behavior in un-timed loop inside of timed loop.
# 18. Registering stamps.
# .
# .
# N+1. Automate the shortcut building.
#
#
#               .. to DONE...
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