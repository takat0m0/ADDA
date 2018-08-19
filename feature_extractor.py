# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import ConvLayer, BNLayer, LinearLayer, Layers, flatten

class FeatureExtractor(Layers):
    def __init__(self, name_scope, fv_dim, layers = None):
        self.fv_dim = fv_dim
        if layers is None:
            layers = self.make_layers(name_scope)
        super().__init__(name_scope, layers)
        
    def make_layers(self, name_scope):
        layers = {'conv1': ConvLayer('{}_conv1'.format(name_scope),
                                     16, 5, 5, 2),
                  'bn1': BNLayer('{}_bn1'.format(name_scope)),
                  'conv2': ConvLayer('{}_conv2'.format(name_scope),
                                     32, 5, 5, 2),
                  'bn2': BNLayer('{}_bn2'.format(name_scope)),
                  'fc': LinearLayer('{}_fc'.format(name_scope), self.fv_dim)
        }
        return layers
        
    def copy(self, name_scope):
        tmp = {_:self[_].copy('{}_{}'.format(name_scope, _)) for _ in self}
        return FeatureExtractor(name_scope, self.fv_dim, tmp)
    
    def __call__(self, figs, is_training = True):

        h = figs
        h = self['conv1'](h, is_training)
        h = self['bn1'](h, is_training)
        h = tf.nn.relu(h)
        h = self['conv2'](h, is_training)
        h = self['bn2'](h, is_training)
        h = tf.nn.relu(h)
        
        h = flatten(h)
        h = self['fc'](h, is_training)
        
        return h

if __name__ == u'__main__':
    fe = FeatureExtractor(u'featuer_extract')
    z = tf.placeholder(tf.float32, [None, 32, 32, 3])
    h = fe(z, True)
    print(h)
    h = fe(z, False)
    copied = fe.copy(u'hogehoge')
    print(fe)
    print(copied)
    h = copied(z, True)
    print(h)
    print(fe.get_variables())
    print(copied.get_variables())    
