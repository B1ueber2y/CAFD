import numpy as np
import pandas as pd
import math
import tensorflow as tf

from tensorflow.python.framework import ops

def read_csv(path):
    df = pd.read_csv(path, sep=',')
    df = df[df.columns[1:]]
    data = df.as_matrix()
    return data

def save_csv(data, path):
    if type(data) == list:
        data = np.asarray(data)
    assert(type(data) == np.ndarray or type(data) == pd.core.frame.DataFrame)
    if type(data) == np.ndarray:
        col = [('X' + str(k)) for k in range(len(data[0]))]
        save_data = pd.DataFrame(data, columns = col)
    save_data.to_csv(path, sep=',')

def weight_variable(shape,name_var):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name_var, shape, tf.float32)

def bias_variable(shape,name_var):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name_var, shape, tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name_ = 'aaa'):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name_)


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                        decay=self.momentum,
                        updates_collections=None,
                        epsilon=self.epsilon,
                        scale=True,
                        is_training=train,
                        scope=self.name)



def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                    strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                     tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                                        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

