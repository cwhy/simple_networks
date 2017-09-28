import tensorflow as tf
import tensorflow.contrib.layers as cl
from typing import Tuple, Union


def xavier_init(size: Tuple[int, int]):
    in_dim = float(size[0])
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.truncated_normal(shape=size, stddev=xavier_stddev)


def flatten(_input: tf.Tensor):
    print('Unfinished')
    dim = tf.reduce_prod(tf.shape(_input)[1:])
    return tf.reshape(_input, [-1, dim])


def simple_net(_in, n_in, n_out, n_hidden=128):
    _W1 = tf.get_variable('W1', initializer=xavier_init((n_in, n_hidden)))
    _b1 = tf.get_variable('b1', initializer=tf.zeros(shape=[n_hidden]))

    _W2 = tf.get_variable('W2', initializer=xavier_init((n_hidden, n_out)))
    _b2 = tf.get_variable('b2', initializer=tf.zeros(shape=[n_out]))

    _h1 = tf.nn.relu(tf.matmul(_in, _W1) + _b1)
    _logit = tf.matmul(_h1, _W2) + _b2
    return _logit


def layer_dense(_in, n_out, scope_name):
    with tf.variable_scope(scope_name):
        _in_dim = int(_in.get_shape()[-1])
        _W = tf.get_variable('W', shape=(_in_dim, n_out),
                             initializer=cl.xavier_initializer(uniform=False))
        _b = tf.get_variable('b', initializer=tf.zeros(shape=[n_out]))
    return tf.matmul(_in, _W) + _b


def dense_net(z, n_units,
              activation_fn=tf.nn.relu,
              drop_out: Union[None, float] = None,
              batch_norm=False,
              is_train=False):
    _flow = z
    for i, n in enumerate(n_units[:-1]):
        _flow = activation_fn(layer_dense(_flow, n, 'fc' + str(i)))
        if batch_norm:
            _flow = tf.layers.batch_normalization(_flow, training=is_train)
        if drop_out is not None:
            _flow = tf.layers.dropout(_flow, rate=drop_out, training=is_train)
    _flow = layer_dense(_flow, n_units[-1], 'fcFinal')
    return _flow


def le_conv(x__: tf.Tensor, n_out: int):
    net = tf.layers.conv2d(x__, 20, 5, activation=tf.nn.relu, name='conv1')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool1')
    net = tf.layers.conv2d(net, 50, 5, activation=tf.nn.relu, name='conv2')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool2')
    net = cl.flatten(net)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    _logits = net
    return _logits


def le_conv_tune(x__: tf.Tensor, n_out: int,
                 activation_fn=tf.nn.relu,
                 drop_out: Union[None, float] = None,
                 batch_norm=False,
                 is_train=True
                 ):
    net = tf.layers.conv2d(x__, 20, 5, activation=activation_fn, name='conv1')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool1')
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.layers.conv2d(net, 50, 5, activation=activation_fn, name='conv2')
    net = tf.layers.max_pooling2d(net, 2, 1, name='pool2')
    net = cl.flatten(net)
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    if drop_out is not None:
        net = tf.layers.dropout(net, rate=drop_out, training=is_train)
    _logits = net
    return _logits


def le_conv_tune_64(x__: tf.Tensor, n_out: int,
                    activation_fn=tf.nn.relu,
                    drop_out: Union[None, float] = None,
                    batch_norm=False,
                    is_train=True
                    ):
    net = tf.layers.conv2d(x__, 16, 5, activation=activation_fn, name='conv1')
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool1')
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    net = tf.layers.conv2d(net, 32, 5, activation=activation_fn, name='conv2')
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool2')
    net = tf.layers.conv2d(net, 64, 3, activation=activation_fn, name='conv3')
    net = tf.layers.max_pooling2d(net, 3, 2, name='pool3')
    net = cl.flatten(net)
    if batch_norm:
        net = tf.layers.batch_normalization(net, training=is_train)
    print(net.shape)
    net = tf.layers.dense(net, n_out, activation=None,
                          kernel_initializer=cl.xavier_initializer(uniform=False))
    if drop_out is not None:
        net = tf.layers.dropout(net, rate=drop_out, training=is_train)
    _logits = net
    return _logits
