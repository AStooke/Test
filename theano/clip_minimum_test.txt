

import numpy as np
import theano
import theano.tensor as T
import tensorflow as tf

SIZE = 10

# NUMPY

np_ratios = 0.4 * np.random.randn(SIZE).astype('float32') + 1
np_ratios = np.clip(np_ratios, 0.1, 100)
np_advs = np.random.randn(SIZE).astype('float32')
np_clip_param = 0.2
np_surr1 = np_ratios * np_advs
np_clip_ratios = np.clip(np_ratios, 1. - np_clip_param, 1 + np_clip_param)
np_surr2 = np_clip_ratios * np_advs
np_surr = np.minimum(np_surr1, np_surr2)
np_loss = - np.sum(np_surr)

print("np_advs:\n", np_advs)
print("np_ratios:\n", np_ratios)
print("np_clip_ratios:\n", np_clip_ratios)
print("np_surr:\n", np_surr)


# THEANO

th_ratios = T.vector('ratios')
th_advs = T.vector('th_advs')
th_clip_param = T.scalar('th_clip_param')
th_surr1 = th_ratios * th_advs
th_loss1 = - T.sum(th_surr1)
th_grad1 = T.grad(th_loss1, wrt=th_advs)
th_clip_ratios = T.clip(th_ratios, 1 - th_clip_param, 1 + th_clip_param)
th_surr2 = th_clip_ratios * th_advs
th_loss2 = - T.sum(th_surr2)
th_grad2 = T.grad(th_loss2, wrt=th_advs)
th_surr = T.minimum(th_surr1, th_surr2)
th_loss = - T.sum(th_surr)
th_grad = T.grad(th_loss, wrt=th_advs)

th_f = theano.function(
    inputs=[th_ratios, th_advs, th_clip_param],
    outputs=[th_surr1, th_surr2, th_surr, th_grad1, th_grad2, th_grad]
)

th_surr1_out, th_surr2_out, th_surr_out, th_grad1_out, th_grad2_out, th_grad_out = \
    th_f(np_ratios, np_advs, np_clip_param)


print("\nth_surr1:\n", th_surr1_out)
print("th_surr2:\n", th_surr2_out)
print("th_surr:\n", th_surr_out)
print("\nth_grad1:\n", th_grad1_out)
print("th_grad2:\n", th_grad2_out)
print("th_grad:\n", th_grad_out)


# Tensorflow
tf_ratios = tf.placeholder(dtype=tf.float32, shape=[None])
tf_advs = tf.placeholder(dtype=tf.float32, shape=[None])
tf_clip_param = tf.placeholder(dtype=tf.float32, shape=[])
tf_surr1 = tf_ratios * tf_advs
tf_loss1 = - tf.reduce_sum(tf_surr1)
tf_grad1 = tf.gradients(tf_loss1, [tf_advs])
tf_clip_ratios = tf.clip_by_value(tf_ratios, 1. - tf_clip_param, 1. + tf_clip_param)
tf_surr2 = tf_clip_ratios * tf_advs
tf_loss2 = - tf.reduce_sum(tf_surr2)
tf_grad2 = tf.gradients(tf_loss2, [tf_advs])
tf_surr = tf.minimum(tf_surr1, tf_surr2)
tf_loss = - tf.reduce_sum(tf_surr)
tf_grad = tf.gradients(tf_loss, [tf_advs])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf_surr1_out, tf_surr2_out, tf_surr_out, tf_grad1_out, tf_grad2_out, tf_grad_out = \
        sess.run([tf_surr1, tf_surr2, tf_surr, tf_grad1, tf_grad2, tf_grad],
            feed_dict={tf_ratios: np_ratios, tf_advs: np_advs, tf_clip_param: np_clip_param})

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
