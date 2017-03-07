import ipdb
from timeit import default_timer as timer

import numpy as np
import theano
import theano.tensor as T

import synkhronos as synk


def make_data(input_shape, batch_size):
    x_data = np.random.randn(batch_size, *input_shape).astype('float32')
    y_data = np.random.randint(low=0, high=5, size=batch_size, dtype='int32')
    return x_data, y_data


def build_mlp(input_var=None, target=None):

    w_1 = theano.shared(np.random.randn(28 * 28, 800).astype('float32'), name='w_1', target=target)
    b_1 = theano.shared(np.random.randn(800).astype('float32'), name='b_1', target=target)
    o_1 = T.nnet.relu(input_var.dot(w_1) + b_1)

    w_2 = theano.shared(np.random.randn(800, 800).astype('float32'), name='w_2', target=target)
    b_2 = theano.shared(np.random.randn(800).astype('float32'), name='b_2', target=target)
    o_2 = T.nnet.relu(o_1.dot(w_2) + b_2)

    w_3 = theano.shared(np.random.randn(800, 10).astype('float32'), name='w_3', target=target)
    b_3 = theano.shared(np.random.randn(10).astype('float32'), name='b_3', target=target)
    o_3 = T.nnet.softmax(o_2.dot(w_3) + b_3)

    params = [w_1, b_1, w_2, b_2, w_3, b_3]

    return o_3, params



def main():

    B_SIZE = 10000
    MID = B_SIZE // 2

    synk.fork()
    import lasagne

    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    prediction, params = build_mlp(input_var)

    # prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # params = lasagne.layers.get_all_params(network, trainable=True)



    grads = theano.grad(loss, wrt=params)
    flat_grad = T.concatenate(list(map(T.flatten, grads)))

    f_loss = synk.function([input_var, target_var], loss, collect_modes=[None], reduce_ops="sum")
    f_grad = synk.function([input_var, target_var], flat_grad, collect_modes=['reduce'])
    # f_loss = theano.function([input_var, target_var], loss)
    # f_grad = theano.function([input_var, target_var], flat_grad)


    synk.distribute()

    x_data, y_data = make_data([28 * 28], B_SIZE)
    # x_data, y_data = make_data([1, 28, 28], B_SIZE)

    loss_1 = f_loss(x_data, y_data)
    grad_1 = f_grad(x_data, y_data)

    x_shmem, y_shmem = f_loss.get_input_shmems()
    x_dat_sh = x_shmem[:B_SIZE]
    y_dat_sh = y_shmem[:B_SIZE]
    x_data_1 = x_data[:MID]
    x_data_2 = x_data[MID:]
    y_data_1 = y_data[:MID]
    y_data_2 = y_data[MID:]

    ITERS = 10
    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss.as_theano(x_data_1, y_data_1)
        loss_j = f_loss.as_theano(x_data_2, y_data_2)
    loss_time = timer() - t0
    print("theano loss_time: ", loss_time)

    t0 = timer()
    for _ in range(ITERS):
        grad_i = f_grad.as_theano(x_data_1, y_data_1)
        grad_j = f_grad.as_theano(x_data_2, y_data_2)
    grad_time = timer() - t0
    print("theano grad_time: ", grad_time)


    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss(x_dat_sh, y_dat_sh)
    loss_time = timer() - t0
    print("synk shmem loss_time: ", loss_time)

    t0 = timer()
    for _ in range(ITERS):
        grad_i = f_grad(x_dat_sh, y_dat_sh)
    grad_time = timer() - t0
    print("synk shmem grad_time: ", grad_time)

    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss(x_data, y_data)
    loss_time = timer() - t0
    print("synk new input loss_time: ", loss_time)

    t0 = timer()
    for _ in range(ITERS):
        grad_i = f_grad(x_data, y_data)
    grad_time = timer() - t0
    print("synk new input grad_time: ", grad_time)

    # ipdb.set_trace()

if __name__ == '__main__':
    main()



