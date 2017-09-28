import tensorflow as tf
import matplotlib.pyplot as plt
import lib.networks as nets
import numpy as np
import datasets.mnist as mnist

data = mnist.train
mb_size = 500

is_train = tf.placeholder(tf.bool, shape=())
x = tf.placeholder(tf.float32, shape=[None] + list(data.dim_x))
# _flow = nets.cl.flatten(x)
y = tf.placeholder(tf.float32, shape=[None] + list(data.dim_y))
print(x.shape)

module_size = 16


def get_module(_in, init=False):
    with tf.variable_scope('module', reuse=init):
        _flow = tf.tanh(_in)
        return nets.dense_net(_flow, [module_size],
                              batch_norm=False,
                              activation_fn=tf.tanh,
                              drop_out=0.3,
                              is_train=is_train)


def get_layer(_name, _in, out_dim, init=False):
    with tf.variable_scope(f'in_{_name}'):
        _flow = nets.dense_net(_in, [module_size])
    _c = get_module(_flow, init=init)
    with tf.variable_scope(f'out_{_name}'):
        return nets.dense_net(_c, [out_dim])


module_inited = False
s_c_e_w_l = tf.nn.sigmoid_cross_entropy_with_logits
losses = []
y_hat_logitss = []
n_agents = 4
for subnet in range(n_agents):
    _flow = nets.cl.flatten(x)
    layer = 0
    for out_dim in [64] + [64]*subnet + [data.dim_y[0]]:
        layer_name = f'{subnet}_{layer}'
        _logits = get_layer(layer_name, _flow, out_dim, module_inited)
        if not module_inited:
            module_inited = True
        layer += 1
    y_hat_logitss.append(_logits)
    loss = tf.reduce_mean(s_c_e_w_l(logits=_logits, labels=y))
    losses.append(loss)

module_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='module')
module_var_dict = {v.name: v for v in module_vars}
print(module_var_dict)
saver = tf.train.Saver(module_var_dict)
y_hat = tf.sigmoid(sum(y_hat_logitss) / n_agents)
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
data_test = mnist.test
data_feed_test = {x: data_test.x,
                  y: data_test.y,
                  is_train: False}
see = False
print(data_test.n_samples)
for ___ in range(1000):
    sess.run(tf.global_variables_initializer())
    if val_W is not None:
        print('rd')
        with tf.variable_scope('module', reuse=True):
            norm_W = np.sqrt(np.mean(np.square(val_W)))
            sess.run(tf.get_variable('fcFinal/W').assign(val_W/norm_W))
    else:
        pass
        # saver.restore(sess, "mnist_16")

    for _ in range(100):
        if see and not (_ % 7):
            with tf.variable_scope('module', reuse=True):
                val_W = sess.run(tf.get_variable('fcFinal/W'))
            plt.matshow(val_W)
            plt.colorbar()
            plt.show(block=False)
        for _ in range(2000):
            _batch = data.sample(mb_size)
            _data_feed = {x: _batch.x, y: _batch.y, is_train: True}
            sess.run(train, feed_dict=_data_feed)
        val_n_wrong_tr = sess.run(n_wrong, feed_dict=_data_feed)
        val_n_wrong_tst = sess.run(n_wrong, feed_dict=data_feed_test)
        print(val_n_wrong_tr, val_n_wrong_tst)
    with tf.variable_scope('module', reuse=True):
        val_W = sess.run(tf.get_variable('fcFinal/W'))
    if ___ % 10 == 0:
        save_path = saver.save(sess, f"mnist_16")
        print("Model saved in file: %s" % save_path)
input()
