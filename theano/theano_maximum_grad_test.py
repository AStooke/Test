

import numpy as np
import theano
import theano.tensor as T
import tensorflow as tf

SIZE = 10

# NUMPY

np_ratios1 = 0.4 * np.random.randn(SIZE).astype('float32') + 1
np_ratios1 = np.clip(np_ratios1, 0.1, 100)
np_ratios2 = 0.4 * np.random.randn(SIZE).astype('float32') + 1
np_ratios2 = np.clip(np_ratios2, 0.1, 100)
np_advs = np.random.randn(SIZE).astype('float32')
np_surr1 = np_ratios1 * np_advs
np_surr2 = np_ratios2 * np_advs
np_surr = np.maximum(np_surr1, np_surr2)
np_loss = - np.sum(np_surr)

print("np_advs:\n", np_advs)
print("np_ratios1:\n", np_ratios1)
print("np_ratios2:\n", np_ratios2)
print("np_surr:\n", np_surr)


# THEANO

th_ratios1 = T.vector('ratios1')
th_ratios2 = T.vector('ratios2')
th_advs = T.vector('th_advs')
th_surr1 = th_ratios1 * th_advs
th_loss1 = - T.sum(th_surr1)
th_grad1 = T.grad(th_loss1, wrt=th_advs)
th_surr2 = th_ratios2 * th_advs
th_loss2 = - T.sum(th_surr2)
th_grad2 = T.grad(th_loss2, wrt=th_advs)
th_surr = T.maximum(th_surr1, th_surr2)
th_loss = - T.sum(th_surr)
th_grad = T.grad(th_loss, wrt=th_advs)

th_f = theano.function(
    inputs=[th_ratios1, th_ratios2, th_advs],
    outputs=[th_surr1, th_surr2, th_surr, th_grad1, th_grad2, th_grad]
)

th_surr1_out, th_surr2_out, th_surr_out, th_grad1_out, th_grad2_out, th_grad_out = \
    th_f(np_ratios1, np_ratios1, np_advs)


print("\nth_surr1:\n", th_surr1_out)
print("th_surr2:\n", th_surr2_out)
print("th_surr:\n", th_surr_out)
print("\nth_grad1:\n", th_grad1_out)
print("th_grad2:\n", th_grad2_out)
print("th_grad:\n", th_grad_out)


# Tensorflow
tf_ratios1 = tf.placeholder(dtype=tf.float32, shape=[None])
tf_ratios2 = tf.placeholder(dtype=tf.float32, shape=[None])
tf_advs = tf.placeholder(dtype=tf.float32, shape=[None])
tf_surr1 = tf_ratios1 * tf_advs
tf_loss1 = - tf.reduce_sum(tf_surr1)
tf_grad1 = tf.gradients(tf_loss1, [tf_advs])
tf_surr2 = tf_ratios2 * tf_advs
tf_loss2 = - tf.reduce_sum(tf_surr2)
tf_grad2 = tf.gradients(tf_loss2, [tf_advs])
tf_surr = tf.maximum(tf_surr1, tf_surr2)
tf_loss = - tf.reduce_sum(tf_surr)
tf_grad = tf.gradients(tf_loss, [tf_advs])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_surr1_out, tf_surr2_out, tf_surr_out, tf_grad1_out, tf_grad2_out, tf_grad_out = \
        sess.run([tf_surr1, tf_surr2, tf_surr, tf_grad1, tf_grad2, tf_grad],
            feed_dict={tf_ratios1: np_ratios1, tf_ratios2: np_ratios1, tf_advs: np_advs})

print("\ntf_surr1:\n", tf_surr1_out)
print("tf_surr2:\n", tf_surr2_out)
print("tf_surr:\n", tf_surr_out)
print("\ntf_grad1:\n", tf_grad1_out)
print("tf_grad2:\n", tf_grad2_out)
print("tf_grad:\n", tf_grad_out)


assert np.allclose(th_surr1_out, tf_surr1_out)
assert np.allclose(th_surr2_out, tf_surr2_out)
assert np.allclose(th_surr_out, tf_surr_out)
assert np.allclose(th_grad1_out, tf_grad1_out)
assert np.allclose(th_grad2_out, tf_grad2_out)
assert np.allclose(th_grad_out, tf_grad_out)

print("\npassed all tests\n")
