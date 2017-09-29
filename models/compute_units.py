import tensorflow as tf
import models.networks as nets
from typing import Callable


def get_normal_module(module_size: int,
                      is_train: tf.Tensor,
                      name: str = 'module') -> Callable[[tf.Tensor, bool], tf.Tensor]:
    def __compute_unit(_in: tf.Tensor, init: bool = False) -> tf.Tensor:
        with tf.variable_scope(name, reuse=init):
            _flow = tf.tanh(_in)
            return nets.dense_net(_flow, [module_size],
                                  batch_norm=False,
                                  activation_fn=tf.tanh,
                                  drop_out=0.3,
                                  is_train=is_train)

    return __compute_unit


def simple_modular_layer(_name: str,
                         module_size: int,
                         compute_unit: Callable[[tf.Tensor, bool], tf.Tensor],
                         _in: tf.Tensor,
                         out_dim: int,
                         init: bool = False) -> tf.Tensor:
    with tf.variable_scope(f'in_{_name}'):
        _flow = nets.dense_net(_in, [module_size])
    _c = compute_unit(_flow, init=init)
    with tf.variable_scope(f'out_{_name}'):
        return nets.dense_net(_c, [out_dim])
