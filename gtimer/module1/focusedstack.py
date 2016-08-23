"""
LIFO list.
Can move the focus (current item of interest) along the list.
Always returns the current (new) focus after action.
"""


class FocusedStack(object):

    def __init__(self, obj_class):
        self.stack = []
        self.focus = None
        self.focus_index = None
        self.obj_class = obj_class

    def create_next(self, *args, **kwargs):
        self.stack.append(self.obj_class(*args, **kwargs))
        return self.focus_last()

    def remove_last(self):
        try:
            self.stack.pop()  # Might want to return the last item.
        except IndexError:
            pass
        finally:
            return self.focus_last()

    def focus_backward(self):
        if self.focus_index > 0:
            self.focus_index -= 1
            self.focus = self.stack[self.focus_index]
        return self.focus

    def focus_forward(self):
        try:
            self.focus = self.stack[self.focus_index + 1]
            self.focus_index += 1
        except IndexError:
            pass
        finally:
            return self.focus

    def focus_last(self):
        try:
            self.focus = self.stack[-1]
            self.focus_index = len(self.stack) - 1
        except IndexError:
            self.focus = None
            self.focus_index = None
        finally:
            return self.focus

    def focus_root(self):
        try:
            self.focus = self.stack[0]
            self.focus_index = 0
        except IndexError:
            self.focus = None
            self.focus_index = None
        finally:
            return self.focus
