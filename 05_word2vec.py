# -*- coding: utf-8 -*-

import os, random, urllib, zipfile, numpy as np, tensorflow as tf
from collections import Counter

def proc_data(v_size, b_size, skip_wind):
	def download(fn='text8.zip', bytes=31344016):
		path = os.path.join('data','w2v',fn)
		if os.path.exists(path): return path
		urllib.urlretrieve('http://mattmahoney.net/dc/'+fn,path)
		assert os.stat(path).st_size == bytes
		return path
	def read_data(path):
		with zipfile.ZipFile(path) as f:
			return tf.compat.as_str(f.read(f.namelist()[0])).split()
	def build_vocab(words, v_size):
		w2i, i2w = {}, {}
		cnt = [('UNK',-1)]
		cnt.extend(Counter(words).most_common(v_size-1))
		for i,(w,_) in enumerate(cnt): w2i[w],i2w[i] = i,w
		return w2i,i2w
	def gen_sample(inds, skip_wind):
		for i,w in enumerate(inds):
			c = random.randint(1,skip_wind)
			for t in inds[max(0,i-c):i]: yield w,t
			for t in inds[i+1:i+c+1]: yield w,t
	def get_batch(iterator, b_size):
		while True:
			cs = np.zeros(b_size,dtype=np.int32)
			ts = np.zeros([b_size,1])
			for i in xrange(b_size):
				cs[i],ts[i] = next(iterator)
			yield cs,ts
	words = read_data(download())
	w2i,i2w = build_vocab(words,v_size)
	inds = [w2i[w] if w in w2i else 0 for w in words]
	return get_batch(gen_sample(inds,skip_wind),b_size)

class SkipGram:
	def __init__(self, v_size=50000, e_size=128, b_size=128, n_neg=64, lr=1.):
		self.v_size = v_size
		self.e_size = e_size
		self.b_size = b_size
		self.n_neg = n_neg
		self.lr = lr
		self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
		self.build_graph()
	def build_graph(self):
		with tf.name_scope('data'):
			self.cw = tf.placeholder(tf.int32,shape=[self.b_size],name='cw')
			self.tw = tf.placeholder(tf.int32,shape=[self.b_size,1],name='tw')
		with tf.name_scope('embed'):
			matrix = tf.Variable(tf.random_uniform([self.v_size,self.e_size],-1.0,1.0),name='matrix')
		with tf.name_scope('loss'):
			embed = tf.nn.embedding_lookup(matrix,self.cw,name='embed')
			nce_w = tf.Variable(tf.truncated_normal([self.v_size,self.e_size],stddev=1.0/(self.e_size**0.5)),name='nce_w')
			nce_b = tf.Variable(tf.zeros([self.v_size]),name='nce_b')
			self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_w,biases=nce_b,labels=self.tw,
				inputs=embed,num_sampled=self.n_neg,num_classes=self.v_size),name='loss')
		self.opt = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
		with tf.name_scope('summaries'):
			tf.summary.scalar('loss',self.loss)
			tf.summary.histogram('histogram loss',self.loss)
			self.summary = tf.summary.merge_all()
	def train(self, skip_win=1, train_steps=10000, n_print=2000, dirn='data/w2v'):
		saver = tf.train.Saver()
		batch_gen = proc_data(self.v_size,self.b_size,skip_win)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			ckpt = tf.train.get_checkpoint_state(os.path.join(dirn,'checkpoint'))
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
			loss_, writer = 0, tf.summary.FileWriter(os.path.join(dirn,'graph'),sess.graph)
			initial_step = self.global_step.eval()
			for i in xrange(initial_step,initial_step+train_steps):
				cs,ts = next(batch_gen)
				_,l,s = sess.run([self.opt,self.loss,self.summary],feed_dict={self.cw:cs,self.tw:ts})
				loss_ += l; writer.add_summary(s,global_step=i)
				if (i+1)%n_print == 0: 
					print loss_; loss_ = 0
					saver.save(sess,os.path.join(dirn,'checkpoint','skip-gram'),i)
			writer.close()

if __name__ == '__main__':
	sg = SkipGram()
	sg.train()
