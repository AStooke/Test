
import numpy as np
import os

GPU = True
if GPU:
    os.environ['THEANO_FLAGS'] = "contexts=dev0->cuda0;dev1->cuda1"
import theano
import theano.tensor as T

INPUT_SIZE = 10
HIDDEN_SIZE = 50
BATCH_SIZE = 100
NUM_LAYERS = 1


def build_mlp(input_var, target_num, n_layers):

    if target_num is None:
        target = "cpu"
        t = "cpu"
    else:
        t = str(target_num)
        target = "dev" + t
    w_1 = theano.shared(np.random.randn(INPUT_SIZE, HIDDEN_SIZE).astype('float32'), name='w' + t + '_1', target=target)
    b_1 = theano.shared(np.random.randn(HIDDEN_SIZE).astype('float32'), name='b' + t + '_1', target=target)
    o_1 = T.nnet.relu(input_var.dot(w_1) + b_1)

    o_prev = o_1
    for i in range(n_layers):
        i_str = '_' + str(i)
        w_i = theano.shared(np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE).astype('float32'), name='w' + t + i_str, target=target)
        b_i = theano.shared(np.random.randn(HIDDEN_SIZE).astype('float32'), name='b' + t + i_str, target=target)
        o_prev = T.nnet.relu(o_prev.dot(w_i) + b_i)

    w_n = theano.shared(np.random.randn(HIDDEN_SIZE, 10).astype('float32'), name='w' + t + '_out', target=target)
    b_n = theano.shared(np.random.randn(10).astype('float32'), name='b' + t + '_out', target=target)
    o_n = T.nnet.softmax(o_prev.dot(w_n) + b_n)

    return o_n


input_var_0 = T.matrix('x0')
input_var_1 = T.matrix('x1')
if GPU:
    pred_0 = build_mlp(input_var_0.transfer('dev0'), 0, NUM_LAYERS)
    pred_1 = build_mlp(input_var_1.transfer('dev1'), 1, NUM_LAYERS)
else:
    pred_0 = build_mlp(input_var_0, None, NUM_LAYERS)
    pred_1 = build_mlp(input_var_1, None, NUM_LAYERS)
f_pred_0 = theano.function([input_var_0], pred_0)
f_pred_1 = theano.function([input_var_1], pred_1)

x_data = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype('float32')
y_data = np.random.randint(low=0, high=5, size=BATCH_SIZE, dtype='int32')

p_0 = f_pred_0(x_data)
p_1 = f_pred_1(x_data)

print(p_0.shape)
