import numpy as np
import tensorflow as tf

import lib.networks as nets
from datasets import DataSet

n_samples = 10000
dim_x = 50
# v = np.random.binomial(1, 0.5, size=[dim_x])
v = np.ones([dim_x])
v[np.random.choice(dim_x, dim_x - 2, replace=False)] = 0
print(v)
dat_X = np.random.binomial(1, 0.5, size=[n_samples, dim_x])

parity_data = DataSet(dat_X, (dat_X @ v % 2)[:, np.newaxis])
parity_data = DataSet(parity_data.x, parity_data.get_y_1hot())

(d_train, d_test) = parity_data.random_split(ratio=0.8)

data = d_train
mb_size = 100

is_train = tf.placeholder(tf.bool, shape=())
x = tf.placeholder(tf.float32, shape=[None] + list(data.dim_x))
z = tf.placeholder(tf.float32, shape=[None] + list(data.dim_x))
# x = tf.cond(is_train, lambda: x + z, lambda: x)
y = tf.placeholder(tf.float32, shape=[None] + list(data.dim_y))
# _flow = nets.cl.flatten(x)
print(x.shape)
y_hat_logits = nets.dense_net(x, [256, 64, 4, data.dim_y[0]],
                              batch_norm=True,
                              activation_fn=tf.tanh,
                              drop_out=0.3,
                              is_train=is_train)
y_hat = tf.sigmoid(y_hat_logits)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.arg_max(y_hat, 1),
                     tf.arg_max(y, 1)), tf.float32))

n_wrong = tf.reduce_sum(
    tf.cast(tf.not_equal(tf.arg_max(y_hat, 1),
                         tf.arg_max(y, 1)), tf.float32))
s_c_e_w_l = tf.nn.sigmoid_cross_entropy_with_logits
loss = tf.reduce_mean(s_c_e_w_l(logits=y_hat_logits, labels=y))
# For batch normalization update
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

marker = 0
epoch = 1
data_test = d_test
data_feed_test = {x: data_test.x,
                  y: data_test.y,
                  is_train: False}

for _ in range(20):
    for _ in range(100):
        (epoch, marker), _batch = data.next_batch(mb_size, (epoch, marker))
        val_z = np.random.normal(0, 0.1, size=[mb_size, dim_x])
        _data_feed = {x: _batch.x, y: _batch.y, z: val_z, is_train: True}
        sess.run(train, feed_dict=_data_feed)
    val_n_wrong_tr = sess.run(n_wrong, feed_dict=_data_feed)
    val_accuracy = sess.run(accuracy, feed_dict=data_feed_test)
    print(val_n_wrong_tr, val_accuracy, f"marker={marker}, epoch={epoch}")
