# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf

from tf_util import LinearLayer, Layers, batch_norm, flatten

class Classifier(Layers):
    def __init__(self, name_scope, class_num, layers = None):
        self.class_num = class_num
        if layers is None:
            layers = self.make_layers(name_scope)
        super().__init__(name_scope, layers)
        
    def make_layers(self, name_scope):
        layers = {'fc1': LinearLayer('{}_fc1'.format(name_scope), 512),
                  'fc2': LinearLayer('{}_fc2'.format(name_scope), self.class_num),
        }
        return layers
        
    def copy(self, name_scope):
        tmp = {_:self[_].copy('{}_{}'.format(name_scope, _)) for _ in self}
        return Classifier(name_scope, self.class_num, tmp)
    
    def __call__(self, figs, is_training = True):

        h = figs
        h = self['fc1'](h, is_training)
        h = tf.nn.relu(h)
        h = self['fc2'](h, is_training)
        logits = h
        probs = tf.nn.softmax(logits)

        return logits, probs

if __name__ == u'__main__':
    c = Classifier(u'Classifier')
    z = tf.placeholder(tf.float32, [None, 1024])
    h = c(z, True)
    print(h)
    h = c(z, False)
    copied = c.copy(u'hogehoge')
    h = copied(z, True)
    print(h)
    print(c.get_variables())
    print(copied.get_variables())    
