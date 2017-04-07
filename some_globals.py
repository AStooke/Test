
sync = None

def give_sync(given_sync):
    global sync
    sync = given_sync

def print_sync():
    print(sync)


class SomeClass(object):

    def print_sync(self):
        print(sync)
