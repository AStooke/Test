from multiprocessing import Process, Value, Array

def f(n, a, name):

    print name
    n.value *= 2
    a.append(n.value)
    # for i in range(len(a)):
    #     a[i] = -a[i]
    


if __name__ == '__main__':
    num = Value('d', 1.)
    arr = Array('i', range(10))
    names = ('bob','sue','joe','alice')
    pro = []
    for i in range(4):
        p = Process(target=f, args=(num, arr, names[i]))
        pro.append(p)
    for i in range(4):
        pro[i].start()
    for i in range(4):
        pro[i].join()


    print num.value
    print arr[:]