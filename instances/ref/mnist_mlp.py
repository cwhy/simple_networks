import tensorflow as tf

import datasets.mnist as mnist
import models.networks as nets

data_train = mnist.train.sample(200)
mb_size = 200

x = tf.placeholder(tf.float32, shape=[None] + list(data_train.dim_x))
y = tf.placeholder(tf.float32, shape=[None] + list(data_train.dim_y))
is_train = tf.placeholder(tf.bool, shape=())
_flow = nets.cl.flatten(x)
# _flow = nets.le_conv_tune(tf.expand_dims(x, -1), 256,
#                           batch_norm=True,
#                           drop_out=None,
#                           is_train=is_train)
y_hat_logits = nets.dense_net(_flow, [64, 64, 64, 64, 64, data_train.dim_y[0]],
                              batch_norm=False,
                              drop_out=0.3,
                              is_train=is_train)
y_hat = tf.sigmoid(y_hat_logits)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.less(y_hat[:, 0], 0.5),
                                           tf.less(y[:, 0], 0.5)),
                                  tf.float32))
n_wrong = tf.reduce_sum(tf.cast(tf.not_equal(tf.less(y_hat[:, 0], 0.5),
                                             tf.less(y[:, 0], 0.5)),
                                tf.float32))
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
data_test = mnist.test
print(data_test.n_samples)
data_feed_test = {x: data_test.x, y: data_test.y, is_train: False}
for _ in range(200):
    for _ in range(70):
        (epoch, marker), _batch = data_train.next_batch(mb_size, (epoch, marker))
        _data_feed = {x: _batch.x, y: _batch.y, is_train: True}
        sess.run(train, feed_dict=_data_feed)
    val_n_wrong_tr = sess.run(n_wrong, feed_dict=_data_feed)
    val_n_wrong = sess.run(n_wrong, feed_dict=data_feed_test)
    print(val_n_wrong_tr, val_n_wrong, f"marker={marker}, epoch={epoch}")
