import tensorflow as tf


def accuracy(y_hat: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.cast(tf.equal(tf.arg_max(y_hat, 1),
                         tf.arg_max(y, 1)), tf.float32))


def n_wrong(y_hat: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(
        tf.cast(tf.not_equal(tf.arg_max(y_hat, 1),
                             tf.arg_max(y, 1)), tf.float32))
