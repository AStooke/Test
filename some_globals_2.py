
from some_globals import print_sync, give_sync
from some_globals import sync as g_sync

print_sync()
give_sync([0, 1, 2])
print_sync()
give_sync(dict(yeah='whoa'))
print_sync()

import some_globals
some_globals.sync = [10, 12, 11]
print_sync()

some_obj = some_globals.SomeClass()
some_obj.print_sync()
some_globals.sync = ["uh-huh, worked!"]
some_obj.print_sync()
