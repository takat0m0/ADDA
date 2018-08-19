# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf

class Layer(object):
    def __call__(self, inputs, is_training = False):
        pass
    def make_variables(self):
        pass
    def copy(self, name_scope):
        pass
    def get_variables(self):
        pass
