import multiprocessing as mp
import time
# import copy


class Yeah():

    def target(self, dict1):
        print(self.yeah)
        dict1["yeah"] = 17
        time.sleep(.1)
        print("done sleeping")

    def main(self):
        manager = mp.Manager()
        dict1 = manager.dict()
        p = mp.Process(target=self.target, args=(dict1,))

        self.yeah = "whoooops"

        p.start()
        p.join()
        print("past join")
        print(type(dict1))
        # print(dir(dict1))
        dict2 = dict1.copy()
        print(type(dict2))





y = Yeah()
y.main()
