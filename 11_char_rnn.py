# -*- coding: utf-8 -*-

import os, random, numpy as np, tensorflow as tf

class CharRNN(object):
	def __init__(self):
		self.vocab = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ !?:;/$%()+-=.,\'"_{}|@#➡'
		self.seq = tf.placeholder(tf.int32,[None,None])
		self.h_dims = [128,256]
		self.n_step = 50
		self.b_size = 64
		self.lr = 0.0003
		self.len_gen = 200
		self.build_model()
	def build_model(self):
		seq = tf.one_hot(self.seq,len(self.vocab))
		layers = [tf.nn.rnn_cell.GRUCell(h_dim) for h_dim in self.h_dims]
		cells = tf.nn.rnn_cell.MultiRNNCell(layers)
		self.in_state = tuple([tf.placeholder_with_default(s,[None,s.shape[1]]) 
			for s in cells.zero_state(tf.shape(seq)[0],dtype=tf.float32)])
		length = tf.reduce_sum(tf.reduce_max(tf.sign(seq),2),1)
		self.output,self.out_state = tf.nn.dynamic_rnn(cells,seq,length,self.in_state)
		self.logits = tf.layers.dense(self.output,len(self.vocab),None)
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits[:,:-1],labels=seq[:,1:])
		self.loss = tf.reduce_sum(loss)
		self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.gstep)
		self.sample = tf.multinomial(tf.exp(self.logits[:,-1]/tf.constant(1.5)),1)[:,0]
	def read_batch(self, path, overlap):
		def read():
			while True:
				for text in [l.strip() for l in open(path).readlines()]:
					text = [self.vocab.index(x)+1 for x in text if x in self.vocab]
					for start in xrange(0,len(text)-self.n_step,overlap):
						yield text[start:start+self.n_step]
		batch = []
		for element in read():
			batch.append(element)
			if len(batch)==self.b_size:
				yield batch; batch = []
		else: yield batch
	def online_infer(self, sess):
		for seed in ['Hillary','A','G','I','M','N','R','T','W','@','.']:
			sent = seed; state = None
			for _ in range(self.len_gen):
				feed = {self.seq:[self.vocab.index(x)+1 for x in text if x in self.vocab]}
				if state is not None:
					for i in xrange(len(state)):
						feed.update({self.in_state[i]:state[i]})
				index,state = sess.run([self.sample,self.out_state],feed)
				sent += ''.join([self.vocab[x-1] for x in index])
			print sent
	def train(self, path, skip_step=20, dirn='data/rnn'):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			ckpt = tf.train.get_checkpoint_state(os.path.join(dirn,'checkpoint'))
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
			data = self.read_batch(path,self.n_step/2)
			i = 0
			while True:
				_,l = sess.run([self.opt,self.loss],{self.seq:next(data)})
				if (i+1)%skip_step == 0:
					saver.save(sess,os.path.join(dirn,'checkpoint'),i)
					print i,l; self.online_infer(sess); i += 1

if __name__ == '__main__':
    CharRNN().train('data/rnn/trump_tweets.txt')
