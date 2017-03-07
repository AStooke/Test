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

    B_SIZE = 1000
    MID = B_SIZE // 2

    # synk.fork()
    import lasagne
    # Before importing theano...contexts="dev0->cuda0;dev1->cuda1"

    input_var_0 = T.matrix('inputs_0').transfer('dev0')
    target_var_0 = T.ivector('targets_0').transfer('dev0')
    prediction_0, params_0 = build_mlp(input_var_0, target='dev0')
    loss_0 = lasagne.objectives.categorical_crossentropy(prediction_0, target_var_0)
    loss_0 = loss_0.mean()
    grads_0 = theano.grad(loss_0, wrt=params_0)
    flat_grad_0 = T.concatenate(list(map(T.flatten, grads_0)))

    input_var_1 = T.matrix('inputs_1').transfer('dev1')
    target_var_1 = T.ivector('targets_1').transfer('dev1')
    prediction_1, params_1 = build_mlp(input_var_1, target='dev1')
    loss_1 = lasagne.objectives.categorical_crossentropy(prediction_1, target_var_1)
    loss_1 = loss_1.mean()
    grads_1 = theano.grad(loss_1, wrt=params_1)
    flat_grad_1 = T.concatenate(list(map(T.flatten, grads_1)))

    # f_loss = synk.function([input_var, target_var], loss, collect_modes=[None], reduce_ops="sum")
    # f_grad = synk.function([input_var, target_var], flat_grad, collect_modes=['reduce'])
    # f_loss = theano.function([input_var, target_var], loss)
    # f_grad = theano.function([input_var, target_var], flat_grad)

    f_pred_0 = theano.function([input_var_0], prediction_0)
    f_pred_1 = theano.function([input_var_1], prediction_1)



    f_loss_0 = theano.function([input_var_0, target_var_0], loss_0)
    f_loss_1 = theano.function([input_var_1, target_var_1], loss_1)
    # f_grad_0 = theano.function([input_var_0, target_var_0], flat_grad_0)

    # all_inputs = [input_var_0, target_var_0, input_var_1, target_var_1]
    # f_loss = theano.function(all_inputs, [loss_0, loss_1])
    # f_grad = theano.function(all_inputs, [flat_grad_0, flat_grad_1])

    print("\n\nf_loss_0:\n")
    theano.printing.debugprint(f_loss_0)
    print("\n\nf_loss_1:\n")
    theano.printing.debugprint(f_loss_1)
    print("\n\nf_pred_0:\n")
    theano.printing.debugprint(f_pred_0)
    print("\n\nf_pred_1:\n")
    theano.printing.debugprint(f_pred_1)
    # print("\n\nf_loss:\n")
    # theano.printing.debugprint(f_loss)
    # synk.distribute()

    x_data, y_data = make_data([28 * 28], B_SIZE)
    # x_data, y_data = make_data([1, 28, 28], B_SIZE)
    pred_0 = f_pred_0(x_data)
    pred_1 = f_pred_1(x_data)

    print("did predictions")
    loss_0 = f_loss_0(x_data, y_data)
    loss_1 = f_loss_1(x_data, y_data)
    # grad_0 = f_grad_0(x_data, y_data)

    # x_shmem, y_shmem = f_loss.get_input_shmems()
    # x_dat_sh = x_shmem[:B_SIZE]
    # y_dat_sh = y_shmem[:B_SIZE]
    x_data_0 = x_data[:MID]
    x_data_1 = x_data[MID:]
    y_data_0 = y_data[:MID]
    y_data_1 = y_data[MID:]

    # all_data = (x_data_0, y_data_0, x_data_1, y_data_1)

    # loss_0, loss_1 = f_loss(*all_data)
    # grad_0, grad_1 = f_grad(*all_data)

    ITERS = 10
    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss_0(x_data_0, y_data_0)
        loss_j = f_loss_0(x_data_1, y_data_1)
    loss_time = timer() - t0
    print("theano loss_0_0 time: ", loss_time)

    t0 = timer()
    for _ in range(ITERS):
        loss_i = f_loss_1(x_data_0, y_data_0)
        loss_j = f_loss_1(x_data_1, y_data_1)
    loss_time = timer() - t0
    print("theano loss_1_1 time: ", loss_time)
    # t0 = timer()
    # for _ in range(ITERS):
    #     grad_i = f_grad_0(x_data_0, y_data_0)
    #     grad_j = f_grad_0(x_data_1, y_data_1)
    # grad_time = timer() - t0
    # print("theano grad_time: ", grad_time)


    # t0 = timer()
    # for _ in range(ITERS):
    #     loss_i = f_loss(x_dat_sh, y_dat_sh)
    # loss_time = timer() - t0
    # print("synk shmem loss_time: ", loss_time)

    # t0 = timer()
    # for _ in range(ITERS):
    #     grad_i = f_grad(x_dat_sh, y_dat_sh)
    # grad_time = timer() - t0
    # print("synk shmem grad_time: ", grad_time)

    # t0 = timer()
    # for _ in range(ITERS):
    #     loss_i, loss_j = f_loss(*all_data)
    # loss_time = timer() - t0
    # print("synk new input loss_time: ", loss_time)

    # t0 = timer()
    # for _ in range(ITERS):
    #     grad_i, grad_j = f_grad(*all_data)
    # grad_time = timer() - t0
    # print("synk new input grad_time: ", grad_time)

    # ipdb.set_trace()

if __name__ == '__main__':
    main()



