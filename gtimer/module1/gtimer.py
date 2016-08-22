# Main file that gets imported.

# Hold global timers here.
import globalholder as g

# Maybe a few functions but maybe better to import them to be more organized.

# Conditional importing of subfunctions.

# Functions that I want the user to use.
from timerfuncs import *

g.create_next_timer('root')


#
#                  ...TO DO...
#
# 1. **Get the child-parent timer relationships working.**
# 2. Test, test
# 3. Auto-focus manager for subfunctions.
# 4. Test, test!
# 5. Self time and all that.
# 6. Allow duplicate.
# 7. Ones the Times class is stabilized, all the reporting
#     stuff that I wanted to do before.
# 8. How to handle multiple separate heap (contexts).
# .
# .
# N. Think about how much of focus manager to expose
#     to the user for manual manipulation.
# N+1. Automate the shortcut building.
#
