# -*- coding: utf-8 -*-

import os, struct, numpy as np, tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tfd = tf.contrib.distributions

x = tf.placeholder(tf.float32,[None,28,28])
y = tf.placeholder(tf.int64,[None])
image = tf.layers.flatten(x)
avg = tf.get_variable('avg',[28*28,10])
std = tf.get_variable('std',[28*28,10])
pr = tfd.MultivariateNormalDiag(tf.ones_like(avg),tf.ones_like(std))
po = tfd.MultivariateNormalDiag(avg,tf.nn.softplus(std))
bias = tf.get_variable('bias',[10])
logit = tf.nn.relu(tf.matmul(image,po.sample())+bias)
dist = tfd.Categorical(logit)
elbo = tf.reduce_mean(dist.log_prob(y))-\
       tf.reduce_mean(tfd.kl_divergence(po,pr))
opt = tf.train.AdamOptimizer(0.001).minimize(-elbo)
p = tf.argmax(tf.nn.softmax(logit),1)
accu = tf.reduce_sum(tf.cast(tf.equal(p,y),tf.float32))

mnist = input_data.read_data_sets('data/mnist'); b_size = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    elbo_ = 0
    for epoch in xrange(30):
        for i in xrange(len(mnist.train.labels)/b_size):
            x_ = mnist.train.images[i*b_size:(i+1)*b_size]
            y_ = mnist.train.labels[i*b_size:(i+1)*b_size]
            elbo_ += sess.run([opt,elbo],{x:x_.reshape([-1,28,28]),y:y_})[1]
        print elbo_; elbo_ = 0
        x_ = mnist.test.images
        y_ = mnist.test.labels
        accu_ = sess.run([accu],{x:x_.reshape([-1,28,28]),y:y_})[0]
        print 1.*accu_/len(mnist.test.labels)
