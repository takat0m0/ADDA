#! -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from .variable_util import get_const_variable, get_rand_variable, get_dim
from .layer import Layer

def linear(name, inputs, out_dim, bias = 0.0):
    in_dim = get_dim(inputs)
    w = get_rand_variable('weight{}'.format(name),
                          [in_dim, out_dim], 1/np.sqrt(in_dim))
    b = get_const_variable('bias{}'.format(name), [out_dim], bias)
    return tf.matmul(inputs, w) + b

class LinearLayer(Layer):

    WEIGHT = 'weight'
    BIAS = 'bias'
    
    def __init__(self, name, out_num):
        self.name = name
        self.out_num = out_num

        self.weights = None
        self.biases = None
        
    def _make_variables(self, in_num):
        weights_shape = [in_num, self.out_num]
        with tf.variable_scope(self.name):
            self.weights = get_rand_variable(self.WEIGHT, weights_shape,
                                             1/np.sqrt(in_num))
            self.biases = get_const_variable(self.BIAS, [self.out_num], 0.0)
            
    def copy(self, copied_name):
        if self.weights is None:
            return None
        
        ret = LinearLayer(copied_name, self.out_num)
        with tf.variable_scope(copied_name):
            ret.weights = tf.Variable(self.weights, name = self.WEIGHT)
            ret.biases = tf.Variable(self.biases, name = self.BIAS)
        return ret
    
    def get_variables(self):
        return [self.weights, self.biases]
    
    def __call__(self, inputs, is_training = False):
        if self.weights is None:
            in_num = get_dim(inputs)
            self._make_variables(in_num)
        return tf.matmul(inputs, self.weights) + self.biases
    
