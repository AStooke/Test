from threading import Event, Thread


class Periodic(object):
    """Periodically run a function with arguments asynchronously in the background
    Period is a float of seconds.
    Don't expect exact precision with timing.
    Threading is used instead of Multiprocessing because we need shared memory
    otherwise changes made by the function to arguments won't be reflected in
    the rest of the script.
    """

    def __init__(self, func, period, args=[], kwargs={}):
        self.period = period
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.seppuku = Event()

    def start(self):
        self.seppuku.clear()
        self.proc = Thread(target=self._doit)
        self.proc.start()

    def stop(self):
        """Nearly immediately kills the Periodic function"""
        self.seppuku.set()
        self.proc.join()

    def _doit(self):
        while True:
            self.seppuku.wait(self.period)
            if self.seppuku.is_set():
                break
            self.func(*self.args, **self.kwargs)


if __name__ == '__main__':
    import time

    def spam(a, b, eggs=None, changeme=None):
        """Example func with useful conventions to be used with Periodic.
        A "static" variable is made by using function attributes.
        Changes made to mutable variables are seen both inside and outside
        of Periodic.
        """

        # "static" variable
        try:
            spam.count += 1
        except AttributeError:
            spam.count = 1

        # change a value in a list
        if isinstance(changeme, list) and len(changeme):
            changeme[0] = 9001

        print(a,b,eggs,changeme, spam.count)

    a = [4,5]  # proof that mutable objects pass nicely
    b = [6,7]  # proof that mutable objects can be changed

    p = Periodic(spam, 1.0, args=(1,2,), kwargs={"eggs":a, "changeme":b})

    print("changeme:", b)

    print("starting")
    p.start()
    print("running")
    time.sleep(2.2)

    # b should have been changed
    print("changeme", b)

    # this change should be seen in the Periodic thread
    a[0] = 9

    time.sleep(3.0)
    print("stopping")
    p.stop()
    print("stopped")
    time.sleep(1.0)
    print("starting again")
    p.start()
    print("running")
    time.sleep(2.0)
    print("stopping")
    p.stop()
    print("stopped")
    print("spam.count: ", spam.count)
