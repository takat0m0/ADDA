# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import LinearLayer, Layers, BNLayer, lrelu, flatten

class Discriminator(Layers):
    def __init__(self, name_scope, layers = None):
        if layers is None:
            layers = self.make_layers(name_scope)
        super().__init__(name_scope, layers)
        
    def make_layers(self, name_scope):
        layers = {'fc1': LinearLayer('{}_fc1'.format(name_scope), 64),
                  'bn1': BNLayer('{}_bn1'.format(name_scope)),
                  'fc2': LinearLayer('{}_fc2'.format(name_scope), 1)
        }
        return layers
        
    def copy(self, name_scope):
        tmp = {_:self[_].copy('{}_{}'.format(name_scope, _)) for _ in self}
        return Discriminator(name_scope, tmp)
    
    def __call__(self, inputs, is_training = True):
        
        h  = inputs

        h = self['fc1'](h, is_training)
        h = lrelu(h)
        h = self['bn1'](h, is_training)
        h = self['fc2'](h, is_training)
        return h

if __name__ == u'__main__':
    dis = Discriminator(u'discriminator')
                        
    imgs = tf.placeholder(tf.float32, [None, 200])
    h = dis(imgs, False)
    h = dis(imgs, True)    
    print(h)
