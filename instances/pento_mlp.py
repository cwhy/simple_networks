import os.path as op

import numpy as np
import tensorflow as tf

import lib.networks as nets
from datasets import DataSet

pento_dir = op.join(op.expanduser('~'), 'Data', 'pento',
                    '64x64_20k_seed_12193885_64patches.npz')
pento_file = np.load(pento_dir)
pento_data = DataSet(pento_file['data'],
                     pento_file['targets'])
pento_data = DataSet(
    x=pento_data.x.reshape((-1, 64, 64)),
    y=pento_data.get_y_1hot())

(d_train, d_test) = pento_data.random_split(ratio=0.8)

data = d_train
mb_size = 200

x = tf.placeholder(tf.float32, shape=[None] + list(data.dim_x))
y = tf.placeholder(tf.float32, shape=[None] + list(data.dim_y))
is_train = tf.placeholder(tf.bool, shape=())
# _flow = nets.cl.flatten(x)
print(x.shape)
_flow = nets.le_conv_tune_64(tf.expand_dims(x, -1), 256,
                             batch_norm=True,
                             drop_out=None,
                             is_train=is_train)
y_hat_logits = nets.dense_net(_flow, [256, data.dim_y[0]],
                              batch_norm=True,
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
    train = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

marker = 0
epoch = 1
data_test = d_test
data_feed_test = {x: data_test.x,
                  y: data_test.y,
                  is_train: False}
for _ in range(200):
    for _ in range(400):
        (epoch, marker), _batch = data.next_batch(mb_size, (epoch, marker))
        _data_feed = {x: _batch.x, y: _batch.y, is_train: True}
        sess.run(train, feed_dict=_data_feed)
    val_n_wrong_tr = sess.run(n_wrong, feed_dict=_data_feed)
    val_accuracy = sess.run(accuracy, feed_dict=data_feed_test)
    print(val_n_wrong_tr, val_accuracy, f"marker={marker}, epoch={epoch}")
