

# Need to build a big CNN, then write an SGD training function.  Test forward
# and backward propagation.

import numpy as np
import theano
import theano.tensor as T
import theano.gpuarray
import lasagne
import lasagne.layers as L
# import lasagne.layers.dnn as LDNN
import time

EPOCHS = 10
BATCHES = 100
BATCH = 128
DATA = BATCH * BATCHES


def build_cnn(
        n_filters,
        f_sizes,
        strides,
        pads,
        hidden_sizes,
        input_shape,
        input_dtype,
        output_size,
        ):

    print("Building CNN")

    input_var = T.tensor4(name='input', dtype=input_dtype)

    l_in = L.InputLayer(shape=(None, *input_shape), input_var=input_var)

    l_hid = l_in
    for n_f, f_s, stride, pad in zip(n_filters, f_sizes, strides, pads):
        l_hid = L.Conv2DLayer(l_hid,
        # l_hid = LDNN.Conv2DDNNLayer(l_hid,
                                     num_filters=n_f,
                                     filter_size=f_s,
                                     stride=stride,
                                     pad=pad)

    print("conv output shape: ", L.get_output_shape(l_hid))

    for h_size in hidden_sizes:
        l_hid = L.DenseLayer(l_hid, num_units=h_size)

    l_out = L.DenseLayer(l_hid,
                         num_units=output_size,
                         nonlinearity=lasagne.nonlinearities.softmax)

    return l_out, input_var


def build_inference(output_layer, input_var):

    print("Building inference function")
    test_prediction = L.get_output(output_layer, deterministic=True)

    f_inference = theano.function(inputs=[input_var], outputs=test_prediction)

    return f_inference


def build_train(output_layer, input_var, target_var):

    print("Building training function")
    train_prediction = L.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var)
    loss = loss.mean()

    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=1e-3)

    f_train = theano.function(inputs=[input_var, target_var],
                              outputs=loss,
                              updates=updates)

    return f_train


def load_data(input_shape, output_size):

    print("Generating synthetic data")
    x = np.random.randn(DATA, *input_shape).astype(theano.config.floatX)
    y = np.random.randint(low=0, high=output_size - 1, size=DATA).astype("int32")

    return x, y


def main():

    input_shape = (3, 84, 84)
    output_size = 10

    output_layer, input_var = build_cnn(n_filters=[32, 64, 64],
                                        f_sizes=[8, 4, 3],
                                        strides=[4, 2, 1],
                                        pads=[(0, 0), (0, 0), (0, 0)],
                                        hidden_sizes=[512],
                                        input_shape=input_shape,
                                        input_dtype=theano.config.floatX,
                                        output_size=output_size,
                                        )

    f_inference = build_inference(output_layer, input_var)

    target_var = T.ivector('target')
    f_train = build_train(output_layer, input_var, target_var)

    x, y = load_data(input_shape, output_size)

    print("Warming up and running timing...")

    for _ in range(5):
        r = f_inference(x[:BATCH])

    t_0 = time.time()
    for _ in range(EPOCHS):
        for i in range(BATCHES):
            r = f_inference(x[i * BATCH:(i + 1) * BATCH])
    t_1 = time.time()
    print("Ran inference on {} batches in {:.3f} s".format(BATCHES * EPOCHS, t_1 - t_0))

    for _ in range(5):
        r = f_train(x[:BATCH], y[:BATCH])

    t_0 = time.time()
    for _ in range(EPOCHS):
        for i in range(BATCHES):
            r = f_train(x[i * BATCH:(i + 1) * BATCH], y[i * BATCH:(i + 1) * BATCH])
    t_1 = time.time()
    print("Ran training on {} batches in {:.3f} s".format(BATCHES * EPOCHS, t_1 - t_0))


if __name__ == "__main__":
    theano.gpuarray.use("cuda")
    main()
