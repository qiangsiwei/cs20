# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

tfd = tf.contrib.distributions

def make_encoder(data, code_size):
    x = tf.layers.flatten(data)
    x = tf.layers.dense(x,200,tf.nn.relu)
    x = tf.layers.dense(x,200,tf.nn.relu)
    return tfd.MultivariateNormalDiag(
        tf.layers.dense(x,code_size),
        tf.layers.dense(x,code_size,tf.nn.softplus))

def make_prior(code_size):
    return tfd.MultivariateNormalDiag(
        tf.zeros(code_size),
        tf.ones(code_size))

def make_decoder(code, data_shape):
    code = tf.layers.dense(code,200,tf.nn.relu)
    code = tf.layers.dense(code,200,tf.nn.relu)
    logit = tf.layers.dense(code,np.prod(data_shape))
    logit = tf.reshape(logit,[-1]+data_shape)
    return tfd.Independent(tfd.Bernoulli(logit),2)

data = tf.placeholder(tf.float32,[None,28,28])
make_encoder = tf.make_template('encoder',make_encoder)
make_decoder = tf.make_template('decoder',make_decoder)
pr = make_prior(code_size=2)
po = make_encoder(data,code_size=2)
codes = po.sample()
likelihood = make_decoder(codes,[28,28]).log_prob(data)
divergence = tfd.kl_divergence(po,pr)
elbo = tf.reduce_mean(likelihood-divergence)
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)
samples = make_decoder(pr.sample(10),[28,28]).mean()

def plot_codes(ax, codes, labels):
    ax.scatter(codes[:,0],codes[:,1],s=2,c=labels,alpha=0.1)
    ax.set_aspect('equal')
    ax.set_xlim(codes.min()-.1,codes.max()+.1)
    ax.set_ylim(codes.min()-.1,codes.max()+.1)
    ax.tick_params(axis='both',which='both',left='off',bottom='off',labelleft='off',labelbottom='off')

def plot_samples(ax, samples):
    for index,sample in enumerate(samples):
        ax[index].imshow(sample,cmap='gray')
        ax[index].axis('off')

mnist = input_data.read_data_sets('data/mnist')
fig, ax = plt.subplots(nrows=20,ncols=11,figsize=(10,20))
with tf.train.MonitoredSession() as sess:
    for epoch in xrange(20):
        elbo_,codes_,samples_ = sess.run([elbo,codes,samples],{data:mnist.test.images.reshape([-1,28,28])})
        print epoch, elbo_
        ax[epoch,0].set_ylabel('Epoch {}'.format(epoch))
        plot_codes(ax[epoch,0],codes_,mnist.test.labels)
        plot_samples(ax[epoch,1:],samples_)
        for _ in xrange(600):
            sess.run(optimize,{data:mnist.train.next_batch(100)[0].reshape([-1,28,28])})
plt.savefig('data/vae/vae-mnist.png',dpi=300,transparent=True,bbox_inches='tight')
