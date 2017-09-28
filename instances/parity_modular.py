import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import lib.networks as nets
from datasets import DataSet

n_samples = 1000
dim_x = 20
# v = np.random.binomial(1, 0.5, size=[dim_x])
v = np.ones([dim_x])
n_v = 3
v[np.random.choice(dim_x, dim_x - n_v, replace=False)] = 0
print(v)
dat_X = np.random.binomial(1, 0.5, size=[n_samples, dim_x])

parity_data = DataSet(dat_X, (dat_X @ v % 2)[:, np.newaxis])
parity_data = DataSet(parity_data.x, parity_data.get_y_1hot())

(d_train, d_test) = parity_data.random_split(ratio=0.8)

data = d_train
mb_size = 400

is_train = tf.placeholder(tf.bool, shape=())
x = tf.placeholder(tf.float32, shape=[None] + list(data.dim_x))
y = tf.placeholder(tf.float32, shape=[None] + list(data.dim_y))
# _flow = nets.cl.flatten(x)
print(x.shape)


def get_module(_in, init=False):
    with tf.variable_scope('module', reuse=init):
        return nets.dense_net(_in, [16],
                              batch_norm=False,
                              activation_fn=tf.tanh,
                              drop_out=0.5,
                              is_train=is_train)


module_inited = False
s_c_e_w_l = tf.nn.sigmoid_cross_entropy_with_logits
losses = []
y_hat_logitss = []
n_agents = 7
for subnet in range(n_agents):
    with tf.variable_scope(f'in_{subnet}'):
        _flow = nets.dense_net(x, [16, 16])
    _c = get_module(_flow, init=module_inited)
    with tf.variable_scope(f'out_{subnet}'):
        _logits = nets.dense_net(_c, [16, data.dim_y[0]])
    y_hat_logitss.append(_logits)
    loss = tf.reduce_mean(s_c_e_w_l(logits=_logits, labels=y))
    losses.append(loss)
    if not module_inited:
        module_inited = True

module_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module')
module_var_dict = {v.name: v for v in module_vars}
print(module_var_dict)
saver = tf.train.Saver(module_var_dict)
y_hat = tf.sigmoid(sum(y_hat_logitss)/n_agents)
accuracy = tf.reduce_mean(
    tf.cast(tf.equal(tf.arg_max(y_hat, 1),
                     tf.arg_max(y, 1)), tf.float32))

n_wrong = tf.reduce_sum(
    tf.cast(tf.not_equal(tf.arg_max(y_hat, 1),
                         tf.arg_max(y, 1)), tf.float32))
# For batch normalization update
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(sum(losses))
sess = tf.Session()

val_W = None
for ___ in range(2):
    sess.run(tf.global_variables_initializer())
    if val_W is not None:
        print('rd')
        with tf.variable_scope('module', reuse=True):
            sess.run(tf.get_variable('fcFinal/W').assign(val_W))
    else:
        saver.restore(sess, "../storage/mnist_16")
    data_test = d_test
    data_feed_test = {x: data_test.x,
                      y: data_test.y,
                      is_train: False}

    see = False
    for _ in range(10):
        if see:
            with tf.variable_scope('module', reuse=True):
                val_W = sess.run(tf.get_variable('fcFinal/W'))
            plt.matshow(val_W)
            plt.show(block=False)
        for _ in range(200):
            _batch = data.sample(mb_size)
            _data_feed = {x: _batch.x, y: _batch.y, is_train: True}
            sess.run(train, feed_dict=_data_feed)
        val_n_wrong_tr = sess.run(n_wrong, feed_dict=_data_feed)
        val_accuracy = sess.run(accuracy, feed_dict=data_feed_test)
        print(val_n_wrong_tr, val_accuracy, f"iter={_}")
    with tf.variable_scope('module', reuse=True):
        val_W = sess.run(tf.get_variable('fcFinal/W'))
# save_path = saver.save(sess, f"v{n_v}")
# print("Model saved in file: %s" % save_path)
