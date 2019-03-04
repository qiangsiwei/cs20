# -*- coding: utf-8 -*-

'''Seq2Seq with Attention with latest APIs'''

import os, random, logging
import tensorflow as tf
from tensorflow.contrib import layers
tf.logging._logger.setLevel(logging.INFO)

get_fn = lambda fn: os.path.join('data/s2s',fn)

def gen_data():
	with open(get_fn('vocab'),'w') as f:
		f.write('\n'.join(['<S>','</S>','<UNK>']+map(str,range(100))))
	with open(get_fn('input'),'w') as fi,open(get_fn('output'),'w') as fo:
		for i in xrange(10000):
			x = [random.randint(0,100)+3 for _ in range(10)]
			y = [(k+5)%100+3 for k in x]
			fi.write(' '.join(map(str,x))+'\n')
			fo.write(' '.join(map(str,y))+'\n')

class Seq2Seq(object):
	def __init__(self):
		self.vc = get_fn('vocab')
		self.fi = get_fn('input')
		self.fo = get_fn('output')
		self.m_dir = get_fn('model/seq2seq')
		self.b_size = 32
		self.i_max = 30
		self.o_max = 30
		self.e_dim = 100
		self.h_dim = 256
		self.tokens = {'B':0,'E':1,'UNK':2}
	def train(self):
		def seq2seq(features, labels, mode, params):
			x,y = features['x'],features['y']
			b_size = tf.shape(x)[0]
			starts = tf.zeros([b_size],dtype=tf.int64)
			o = tf.concat([tf.expand_dims(starts,1),y],1)
			x_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(x,1)),1)
			o_lens = tf.reduce_sum(tf.to_int32(tf.not_equal(o,1)),1)
			x_embed = layers.embed_sequence(x,vocab_size=self.v_size,embed_dim=self.e_dim,scope='embed')
			o_embed = layers.embed_sequence(o,vocab_size=self.v_size,embed_dim=self.e_dim,scope='embed',reuse=True)
			with tf.variable_scope('embed',reuse=True): embeds = tf.get_variable('embeddings')
			cell = tf.contrib.rnn.GRUCell(num_units=self.h_dim)
			enc_outs,enc_state = tf.nn.dynamic_rnn(cell,x_embed,dtype=tf.float32)
			tr_helper = tf.contrib.seq2seq.TrainingHelper(o_embed,o_lens)
			pr_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
			    embeds,start_tokens=tf.to_int32(starts),end_token=1)
			def decode(helper, scope, reuse=None):
				with tf.variable_scope(scope,reuse=reuse):
					attn = tf.contrib.seq2seq.BahdanauAttention(
						num_units=self.h_dim,memory=enc_outs,
						memory_sequence_length=x_lens)
					cell = tf.contrib.rnn.GRUCell(num_units=self.h_dim)
					attn_cell = tf.contrib.seq2seq.AttentionWrapper(
						cell,attn,attention_layer_size=self.h_dim/2)
					out_cell = tf.contrib.rnn.OutputProjectionWrapper(
						attn_cell,self.v_size,reuse=reuse)
					decoder = tf.contrib.seq2seq.BasicDecoder(
						cell=out_cell,helper=helper,
						initial_state=out_cell.zero_state(
							dtype=tf.float32,batch_size=self.b_size))
					outputs = tf.contrib.seq2seq.dynamic_decode(
						decoder=decoder,output_time_major=False,
						impute_finished=True,maximum_iterations=self.o_max)
					return outputs[0]
			tr_outs = decode(tr_helper,'decode')
			pr_outs = decode(pr_helper,'decode',reuse=True)
			tf.identity(tr_outs.sample_id[0],name='train_pred')
			weights = tf.to_float(tf.not_equal(o[:,:-1],1))
			loss = tf.contrib.seq2seq.sequence_loss(
			    tr_outs.rnn_output,y,weights=weights)
			train_op = layers.optimize_loss(
			    loss,tf.train.get_global_step(),
			    optimizer='Adam',learning_rate=0.001,
			    summaries=['loss','learning_rate'])
			tf.identity(pr_outs.sample_id[0],name='prediction')
			return tf.estimator.EstimatorSpec(
				mode=mode,predictions=pr_outs.sample_id,loss=loss,train_op=train_op)
		def make_input_fn(fi, fo, vocab):
			def input_fn():
				x = tf.placeholder(tf.int64,shape=[None,None],name='x')
				y = tf.placeholder(tf.int64,shape=[None,None],name='y')
				tf.identity(x[0],'x_0')
				tf.identity(y[0],'y_0')
				return {'x':x,'y':y},None
			def sampler():
				i_proc = o_proc = lambda ln,vocab:\
					[vocab.get(w,self.tokens['UNK']) for w in ln.split(' ')]
				while True:
					for li,lo in zip(open(fi).readlines(),open(fo).readlines()):
						yield {'x':i_proc(li,vocab)[:self.i_max-1]+[self.tokens['E']],
							   'y':o_proc(lo,vocab)[:self.o_max-1]+[self.tokens['E']]}
			s = sampler()
			def feed_fn():
				x,y = [],[]; xl,yl = 0,0
				for i in xrange(self.b_size):
				    r = s.next(); x.append(r['x']); y.append(r['y'])
				    xl,yl = max(xl,len(x[-1])),max(yl,len(y[-1]))
				for i in xrange(self.b_size):
				    x[i] += [self.tokens['E']]*(xl-len(x[i]))
				    y[i] += [self.tokens['E']]*(yl-len(y[i]))
				return {'x:0':x,'y:0':y}
			return input_fn, feed_fn
		def get_formatter(keys, vocab):
			r_vocab = {i:w for w,i in vocab.iteritems()}
			to_str = lambda seq:' '.join([r_vocab.get(x,'<UNK>') for x in seq]) 
			return lambda val:'\n'.join(['{0}={1}'.format(key,to_str(val[key])) for key in keys])
		vocab = {w:i for i,w in enumerate(open(self.vc).read().strip().split('\n'))}
		self.v_size = len(vocab)
		est = tf.estimator.Estimator(model_fn=seq2seq,model_dir=self.m_dir)
		input_fn,feed_fn = make_input_fn(self.fi,self.fo,vocab)
		print_x = tf.train.LoggingTensorHook(['x_0','y_0'],every_n_iter=100,
			formatter=get_formatter(['x_0','y_0'],vocab))
		print_y = tf.train.LoggingTensorHook(['prediction','train_pred'],every_n_iter=100,
			formatter=get_formatter(['prediction','train_pred'],vocab))
		est.train(input_fn=input_fn,steps=10000,
			hooks=[tf.train.FeedFnHook(feed_fn),print_x,print_y])

if __name__ == '__main__':
	gen_data()
	Seq2Seq().train()
