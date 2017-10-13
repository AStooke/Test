
import numpy as np
import tensorflow as tf

import time

EPOCHS = 10
BATCHES = 100
BATCH = 128
DATA = BATCH * BATCHES
FORMAT = "NCHW"


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    with tf.device("/gpu:0"):
        w = tf.Variable(initial, dtype=tf.float32)
    return w


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    with tf.device("/gpu:0"):
        b = tf.Variable(initial, dtype=tf.float32)
    return b


def build_cnn(img_shape, output_size, size="small"):
    print("Building CNN")

    if size == "small":
        with tf.device("/gpu:0"):
            input_var = tf.Variable(tf.random_normal(shape=(BATCH, *img_shape),
                                                     dtype=tf.float32),
                                    name="input")

        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([8, 8, 3, 32])
            b_conv1 = bias_variable([32])
            conv1 = tf.nn.conv2d(input_var, W_conv1,
                strides=[1, 1, 4, 4],
                padding='VALID',
                data_format=FORMAT)
            h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b_conv1, data_format=FORMAT))

        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([4, 4, 32, 64])
            b_conv2 = bias_variable([64])
            conv2 = tf.nn.conv2d(h_conv1, W_conv2,
                strides=[1, 1, 2, 2],
                padding='VALID',
                data_format=FORMAT)
            h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b_conv2, data_format=FORMAT))

        with tf.name_scope('conv3'):
            W_conv3 = weight_variable([3, 3, 64, 64])
            b_conv3 = bias_variable([64])
            conv3 = tf.nn.conv2d(h_conv2, W_conv3,
                strides=[1, 1, 1, 1],
                padding='VALID',
                data_format=FORMAT)
            h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b_conv3, data_format=FORMAT))

        conv3_out_size = int(np.prod(h_conv3.get_shape().as_list()[1:]))
        conv3_flat = tf.reshape(h_conv3, [-1, conv3_out_size])
        print("h_conv3 shape: ", h_conv3.get_shape())
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([conv3_out_size, 512])
            b_fc1 = bias_variable([512])
            h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

        with tf.name_scope('fc2'):
            W_fc2 = weight_variable([512, output_size])
            b_fc2 = bias_variable([output_size])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

        trainable_variables = [W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3,
                               W_fc1, b_fc1, W_fc2, b_fc2]

    else:
        raise NotImplementedError

    return logits, input_var, trainable_variables


def build_inference(logits, input_var):
    print("Building inference")
    prediction = tf.argmax(tf.nn.softmax(logits), 1)  # would use the softmax normally
    return prediction


def build_train(logits, input_var, target_var, trainable_variables):
    print("Building training")
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=target_var,
                            logits=logits)

    cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('sgd_optimizer'):
        opt = tf.train.GradientDescentOptimizer(learning_rate=1e-3)
        train_step = opt.minimize(cross_entropy, var_list=trainable_variables)

    return train_step


def load_data(img_shape, output_size):
    print("Generating synthetic data")
    x = np.random.randn(DATA, *img_shape).astype("float32")
    y = np.random.randint(low=0, high=output_size - 1, size=DATA).astype("int32")

    return x, y


def main():

    img_shape = (3, 84, 84)
    # img_shape = (84, 84, 3)
    output_size = 10

    logits, input_var, trainable_variables = build_cnn(img_shape, output_size)

    f_inference = build_inference(logits, input_var)

    # with tf.device("/gpu:0"):
    target_var = tf.Variable(tf.ones([BATCH], dtype=tf.int32), name="target")

    f_train = build_train(logits, input_var, target_var, trainable_variables)

    x, y = load_data(img_shape, output_size)

    print("Warming up and running timing...")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(5):
            r = f_inference.eval()
            # r = sess.run(f_inference, feed_dict={input_var: x[:BATCH]})

        t_0 = time.time()
        for _ in range(EPOCHS):
            for i in range(BATCHES):
                r = f_inference.eval()
                # r = sess.run(f_inference, feed_dict={input_var: x[i * BATCH:(i + 1) * BATCH]})
        t_1 = time.time()
        print("Ran inference on {} batches in {:.3f} s".format(BATCHES * EPOCHS, t_1 - t_0))

        for _ in range(5):
            r = f_train.run()
            # r = sess.run(f_train, feed_dict={input_var: x[:BATCH],
            #                                  target_var: y[:BATCH]})

        t_0 = time.time()
        for _ in range(EPOCHS):
            for i in range(BATCHES):
                r = f_train.run()
                # r = sess.run(f_train, feed_dict={input_var: x[i * BATCH:(i + 1) * BATCH],
                #                                  target_var: y[i * BATCH:(i + 1) * BATCH]})
        t_1 = time.time()
        print("Ran training on {} batches in {:.3f} s".format(BATCHES * EPOCHS, t_1 - t_0))

        # print("getting trace")
        # run_metadata = tf.RunMetadata()
        # _ = sess.run(f_train,
        #            feed_dict={input_var: x[:BATCH], target_var: y[:BATCH]},
        #            options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        #            run_metadata=run_metadata)

        # from tensorflow.python.client import timeline
        # trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        # trace_file = open('timeline.ctf.json', 'w')
        # trace_file.write(trace.generate_chrome_trace_format())
        # print("writing trace")


if __name__ == "__main__":
    main()
