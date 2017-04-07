

class NpArrayHolder(object):

    def __init__(self, data_arr):
        self.data = data_arr

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v
