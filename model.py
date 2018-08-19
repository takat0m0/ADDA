# -*- coding:utf-8 -*-

import os
import sys

import numpy as np
import tensorflow as tf

from feature_extractor import FeatureExtractor
from classifier import Classifier
from discriminator import Discriminator

class Model(object):
    def __init__(self, fv_dim, class_num):

        self.lr = 0.002
        
        # --  placeholder -------
        self.source_figs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.labels = tf.placeholder(tf.float32, [None, class_num])
        
        # -- FeatureExtractor -----
        self.source_fe = FeatureExtractor('source_fe', fv_dim)

        # -- classifier ---
        self.cl = Classifier('classifier', class_num)

        # -- discriminator --
        self.disc = Discriminator('discriminator')
        
    def set_source_model(self):
        # -- classify --
        tmp_fv = self.source_fe(self.source_figs, True)
        logits, probs = self.cl(tmp_fv, True)
        self.obj = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(            
                logits = logits,
                labels = self.labels
            )
        )
        var_list = self.source_fe.get_variables()
        var_list.extend(self.cl.get_variables())
        self.train_source_fe = tf.train.AdamOptimizer(self.lr).minimize(self.obj,
                                                                        var_list = var_list)
        
        # -- for fv --
        self.source_fv = self.source_fe(self.source_figs, False)
        self.source_prob = self.cl(self.source_fv, False)
        
    def init(self, sess):
        var_list = self.source_fe.get_variables()
        var_list.extend(self.cl.get_variables())
        init_op = tf.variables_initializer(var_list = var_list)
        sess.run(init_op)
        
    def set_target(self, sess):
        # -- make target fe ---
        self.target_figs = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.target_fe = self.source_fe.copy('target_fe')

        # -- make target fe loss --
        target_fv = self.target_fe(self.target_figs, True)
        t_logits = self.disc(target_fv, True)
        self.target_obj =  -tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits = t_logits,
                labels = tf.zeros_like(t_logits)
            )
        )

        # -- make adversarial loss ---
        d_logits = self.disc(self.source_fv, True)
        d_obj_true = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(            
                logits = d_logits,
                labels = tf.ones_like(d_logits)
            )
        )
        d_obj_false = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(            
                logits = t_logits,
                labels = tf.zeros_like(t_logits)
            )
        )
        self.disc_obj = d_obj_true + d_obj_false

        # -- make traing_ops ---
        self.train_target_fe  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.target_obj, var_list = self.target_fe.get_variables())
        self.train_discriminator  = tf.train.AdamOptimizer(self.lr, beta1 = 0.5).minimize(self.disc_obj, var_list = self.disc.get_variables())

        # -- target_fv ---
        self.target_fv = self.target_fe(self.target_figs, False)
        self.target_prob = self.cl(self.target_fv, False)
        
        # -- initialize target fe variables --
        var_list = self.target_fe.get_variables()
        init_op = tf.variables_initializer(var_list = var_list)
        sess.run(init_op)
        
    def training_source(self, sess, source_figs):
        _, obj = sess.run([self.train_source_fe, self.obj],
                            feed_dict = {self.source_figs: source_figs})
        return obj
    
    def training_target(self, sess, target_figs):
        _, obj = sess.run([self.train_target_fe, self.target_obj],
                            feed_dict = {self.target_figs: target_figs})
        return obj
    
    def training_discriminator(self, sess, souce_figs, target_figs):
        _, obj = sess.run([self.train_discriminator, self.disc_obj],
                            feed_dict = {self.target_figs: target_figs,
                                         self.source_figs: source_figs})
        return obj
    
    def get_souce_fv_and_prob(self, sess, figs):
        fv, prob = sess.run([self.souce_fv, self.source_prob],
                            feed_dict = {self.souce_figs: figs})
        return fv, prob
    
    def get_target_fv_and_prob(self, sess, target_figs):
        fv, prob = sess.run([self.targe_fv, self.target_prob],
                            feed_dict = {self.target_figs: target_figs})
        return fv, prob
    
if __name__ == u'__main__':
    model = Model(200, 10)
    model.set_source_model()
    with tf.Session() as sess:
        model.init(sess)
        model.set_target(sess)
