#! -*- coding:utf-8 -*-

import tensorflow as tf

from .variable_util import get_const_variable, get_rand_variable
from .layer import Layer

def deconv(name, inputs, out_shape, filter_width, filter_hight, stride):

    # ** NOTICE: weight shape is [hight, width, out_chanel, in_chanel] **
    out_channel = out_shape[-1]
    in_channel = inputs.get_shape()[-1]
    weights_shape =  [filter_hight, filter_width, out_channel, in_channel]
    weights = get_rand_variable(name,weights_shape, 0.02)
    
    biases = get_const_variable(name, [out_shape[-1]], 0.0)
    
    deconved = tf.nn.conv2d_transpose(inputs, weights, output_shape = out_shape,
                                      strides=[1, stride,  stride,  1])
    return tf.nn.bias_add(deconved, biases)

class DeconvLayer(Layer):

    WEIGHT = 'weight'
    BIAS = 'bias'
    
    def __init__(self, name, out_shape, filter_width, filter_height, stride):
        self.name = name
        self.out_shape = out_shape
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride

        self.weights = None
        self.biases = None
        
    def _make_variables(self, in_num):
        weights_shape = [self.filter_height, self.filter_width,
                         self.out_shape[-1], in_num]
        with tf.variable_scope(self.name):
            self.weights = get_rand_variable(self.WEIGHT, weights_shape, 0.02)
            self.biases = get_const_variable(self.BIAS, [self.out_shape[-1]], 0.0)
            
    def copy(self, copied_name):
        if self.weights is None:
            return None
        
        ret = DeconvLayer(copied_name, self.out_shape,
                        self.filter_width, self.filter_height, self.stride)
        with tf.variable_scope(copied_name):
            ret.weights = tf.Variable(self.weights, name = self.WEIGHT)
            ret.biases = tf.Variable(self.biases, name = self.BIAS)
        return ret
    
    def get_variables(self):
        return [self.weights, self.biases]
    
    def __call__(self, inputs, is_training = False):
        if self.weights is None:
            in_num = inputs.get_shape()[-1]
            self._make_variables(in_num)
        deconved = tf.nn.conv2d_transpose(inputs, self.weights,
                                          output_shape = self.out_shape,
                                          strides=[1, self.stride, self.stride,  1])
        return tf.nn.bias_add(deconved, self.biases)
            
if __name__ == '__main__':
    deconv = DeconvLayer('deconv', [10, 128, 128, 32], 3, 3, 2)
    fig = tf.placeholder(tf.float32, [10, 64, 64, 3])
    h = deconv(fig, True)
    h = deconv(fig, False)
    print(h)
    copied = deconv.copy('hoge')
    h = copied(fig, True)
    print(h)
