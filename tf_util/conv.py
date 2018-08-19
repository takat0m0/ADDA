# -*- coding:utf-8 -*-

import tensorflow as tf
from .layer import Layer
from .variable_util import get_const_variable, get_rand_variable

def conv(name, inputs, out_num, filter_width, filter_height, stride):

    # ** NOTICE: weight shape is [height, width, in_chanel, out_chanel] **
    
    in_channel = inputs.get_shape()[-1]
    weights_shape = [filter_height, filter_width, in_channel, out_num]
    weights = get_rand_variable('{}/weight'.format(name), weights_shape, 0.02)

    biases = get_const_variable('{}/bias'.format(name), [out_num], 0.0)
    
    conved = tf.nn.conv2d(inputs, weights,
                          strides=[1, stride,  stride,  1],
                          padding = 'SAME')
    
    return tf.nn.bias_add(conved, biases)

class ConvLayer(Layer):

    WEIGHT = 'weight'
    BIAS = 'bias'
    
    def __init__(self, name, out_num, filter_width, filter_height, stride):
        self.name = name
        self.out_num = out_num
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride

        self.weights = None
        self.biases = None
        
    def _make_variables(self, in_num):
        weights_shape = [self.filter_height, self.filter_width,
                         in_num, self.out_num]
        with tf.variable_scope(self.name):
            self.weights = get_rand_variable(self.WEIGHT, weights_shape, 0.02)
            self.biases = get_const_variable(self.BIAS, [self.out_num], 0.0)
            
    def copy(self, copied_name):
        if self.weights is None:
            return None
        
        ret = ConvLayer(copied_name, self.out_num,
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
        conved = tf.nn.conv2d(inputs, self.weights,
                              strides=[1, self.stride, self.stride,  1],
                              padding = 'SAME')
    
        return tf.nn.bias_add(conved, self.biases)
    
