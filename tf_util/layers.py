# -*- coding:utf-8 -*-

import os
import sys
import tensorflow as tf

class Layers(object):
    def __init__(self, name, layer_dict):
        self.name = name
        self.__layers = layer_dict
        
    def make_variables(self):
        for k in self.__layers:
            self.__layers[k].make_variables()
            
    def __iter__(self):
        return self.__layers.__iter__()

    def __getitem__(self, layer_name):
        return self.__layers[layer_name]
    
    def get_variables(self):
        ret = []
        for k in self.__layers:
            ret.extend(self.__layers[k].get_variables())
        return ret

    def copy(self, name_scope):
        pass
    
    def __call__(self, inputs, is_training):
        pass
