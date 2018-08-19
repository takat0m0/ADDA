# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from .variable_util import get_const_variable
from .layer import Layer

def batch_norm(name, x, decay_rate = 0.99, is_training = True):
    #decay_rate = 0.99
    
    shape = x.get_shape().as_list()
    dim = shape[-1]
    if len(shape) == 2:
        mean, var = tf.nn.moments(x, [0], name = 'moments_bn_{}'.format(name))
    elif len(shape) == 4:
        mean, var = tf.nn.moments(x, [0, 1, 2], name = 'moments_bn_{}'.format(name))

    avg_mean  = get_const_variable('avg_mean_bn_{}'.format(name),
                                   [1, dim], 0.0, False)
    
    avg_var = get_const_variable('avg_var_bn_{}'.format(name),
                                 [1, dim], 1.0, False)
    
    beta  = get_const_variable('beta_bn_{}'.format(name),
                               [1, dim], 0.0)
    gamma = get_const_variable('gamma_bn_{}'.format(name),
                               [1, dim], 1.0)

    if is_training:
        avg_mean_assign_op = tf.assign(avg_mean, decay_rate * avg_mean
                                       + (1 - decay_rate) * mean)
        avg_var_assign_op = tf.assign(avg_var,
                                      decay_rate * avg_var
                                      + (1 - decay_rate) * var)

        with tf.control_dependencies([avg_mean_assign_op, avg_var_assign_op]):
            ret = gamma * (x - mean) / tf.sqrt(1e-6 + var) + beta
    else:
        ret = gamma * (x - avg_mean) / tf.sqrt(1e-6 + avg_var) + beta
        
    return ret

class BNLayer(Layer):

    AVE_MEAN = 'AVE_MEAN'
    AVE_VAR  = 'AVE_VAR'    
    BETA  = 'beta'
    GAMMA = 'gamma'    
    
    def __init__(self, name, decay_rate = 0.99):
        self.name = name
        self.decay_rate = decay_rate

        self.ave_mean = None
        self.ave_var = None
        self.beta = None
        self.gamma = None
        
    def _make_variables(self, dim):
        with tf.variable_scope(self.name):
            self.ave_mean  = get_const_variable(self.AVE_MEAN,
                                                [1, dim], 0.0, False)
    
            self.ave_var = get_const_variable(self.AVE_VAR,
                                              [1, dim], 1.0, False)
    
            self.beta  = get_const_variable(self.BETA,
                                            [1, dim], 0.0)
            self.gamma = get_const_variable(self.GAMMA,
                                            [1, dim], 1.0)            
            
    def copy(self, copied_name):
        ret = BNLayer(copied_name, self.decay_rate)
        with tf.variable_scope(copied_name):
            ret.ave_mean = tf.Variable(self.ave_mean, name = self.AVE_MEAN,
                                       trainable = False)
            ret.ave_var  = tf.Variable(self.ave_var, name = self.AVE_VAR,
                                       trainable = False)
            ret.beta  = tf.Variable(self.beta, name = self.BETA)
            ret.gamma = tf.Variable(self.gamma, name = self.GAMMA)
        return ret
    
    def get_variables(self):
        return [self.beta, self.gamma, self.ave_mean, self.ave_var]
    
    def __call__(self, inputs, is_training = False):
        shape = inputs.get_shape().as_list()
        dim = shape[-1]
        if self.ave_mean is None:
            self._make_variables(dim)
        
        if len(shape) == 2:
            mean, var = tf.nn.moments(inputs, [0], name = '{}_moments'.format(self.name))
        elif len(shape) == 4:
            mean, var = tf.nn.moments(inputs, [0, 1, 2], name = '{}_moments'.format(self.name))

        if is_training:
            ave_mean_assign_op = tf.assign(self.ave_mean, self.decay_rate * self.ave_mean
                                           + (1 - self.decay_rate) * mean)
            ave_var_assign_op = tf.assign(self.ave_var,
                                          self.decay_rate * self.ave_var
                                          + (1 - self.decay_rate) * var)

            with tf.control_dependencies([ave_mean_assign_op, ave_var_assign_op]):
                ret = self.gamma * (inputs - mean) / tf.sqrt(1e-6 + var) + self.beta
        else:
            ret = self.gamma * (inputs - self.ave_mean) / tf.sqrt(1e-6 + self.ave_var) + self.beta
        
        return ret
        


if __name__ == u'__main__':
    x = tf.placeholder(dtype = tf.float32, shape = [None, 10, 10, 3])
    batch_norm(1, x, 0.9, True)
