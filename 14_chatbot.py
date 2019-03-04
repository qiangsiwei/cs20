# -*- coding: utf-8 -*-

import os, random, regex as re
from collections import defaultdict
import numpy as np, tensorflow as tf

config = {'buckets':[(19,19),(28,28),(33,33),(40,43),(50,53),(60,63)]}

class DataProc(object):
	def __init__(self):
		self.lines_fn = 'data/cornell/movie_lines.txt'
		self.convs_fn = 'data/cornell/movie_conversations.txt'
		self.proc_dir = 'data/cornell/processed'
		self.test_size = 25000
		self.sid = {'pad':0,'unk':1,'B':2,'E':3}
	def prepare_data(self):
		def get_lines():
			for l in open(self.lines_fn).read().strip().split('\n'):
				l = l.split(' +++$+++ '); yield l[0],l[4]
		def get_convs():
			for l in open(self.convs_fn).read().strip().split('\n'):
				l = l.split(' +++$+++ '); yield re.findall(r'L\d+',l[-1])
		def QA(id2ln, convs):
			return zip(*[(q,a) for conv in convs for q,a in \
				zip(map(id2ln.get,conv[:-1]),map(id2ln.get,conv[1:]))])
		def prepare_dataset(Qs, As):
			f = [open(os.path.join(self.proc_dir,fn),'w') \
				for fn in ['train.enc','train.dec','test.enc','test.dec']]
			test_ids = random.sample(range(len(Qs)),self.test_size)
			for i in range(len(Qs)):
				fq,fa = (f[2],f[3]) if i in test_ids else (f[0],f[1])
				fq.write(Qs[i]+'\n'); fa.write(As[i]+'\n')
			for f_ in f: f_.close()
		prepare_dataset(*QA(dict(get_lines()),list(get_convs())))
	def tokenize(self, ln):
			return filter(None,re.split(r'([.,!?\"\'-<>:;)(])|\s',re.sub(r'\d','#',\
				re.sub(r'(</?u>|\[|\])','',ln).strip().lower())))
	def sent2id(self, vocab, ln):
		return [vocab.get(w,vocab['<unk>']) for w in self.tokenize(ln)]
	def process_data(self):
		def build_vocab(fn):
			vocab = defaultdict(int)
			for ln in open(os.path.join(self.proc_dir,fn)).read().strip().split('\n'):
				for w in self.tokenize(ln): vocab[w] += 1
			vocab = filter(lambda w:vocab[w]>=2,sorted(vocab,key=vocab.get,reverse=True))
			with open(os.path.join(self.proc_dir,'vocab.'+fn[-3:]),'w') as out:
				out.write('\n'.join(['<pad>','<unk>','<s>','<\s>']+vocab))
			return len(vocab)+4
		def token2id(data, mode):
			vocab = {w:i for i,w in enumerate(open(os.path.join(\
				self.proc_dir,'vocab.'+mode)).read().strip().split('\n'))}
			with open(os.path.join(self.proc_dir,data+'_ids.'+mode),'w') as out:
				for ln in open(os.path.join(self.proc_dir,data+'.'+mode)).read().split('\n'):
					ids = [vocab['<s>']] if mode == 'dec' else []
					ids.extend(self.sent2id(vocab,ln))
					if mode == 'dec': ids.append(vocab['<\s>'])
					out.write(' '.join(str(id_) for id_ in ids)+'\n')
		encv,decv = build_vocab('train.enc'),build_vocab('train.dec')
		[token2id(data,mode) for data in ('train','test') for mode in ('enc','dec')]
		return encv, decv
	def load_data(self, fn_enc, fn_dec):
		encf = open(os.path.join(self.proc_dir,fn_enc),'r')
		decf = open(os.path.join(self.proc_dir,fn_dec),'r')
		enc,dec = encf.readline(),decf.readline()
		buc = [[] for _ in config['buckets']]
		while enc and dec:
			enc_ids,dec_ids = map(int,enc.split()),map(int,dec.split())
			for bid,(enc_max,dec_max) in enumerate(config['buckets']):
				if len(enc_ids)<=enc_max and len(dec_ids)<=dec_max:
					buc[bid].append([enc_ids,dec_ids]); break
			enc,dec = encf.readline(),decf.readline()
		return buc
	def get_batch(self, buc, bid, b_size=1):
		pad = lambda x,size: x+[self.sid['pad']]*(size-len(x))
		enc_in,dec_in = [],[]; enc_size,dec_size = config['buckets'][bid]
		for _ in range(b_size):
			e,d = random.choice(buc)
			dec_in.append(pad(d,dec_size))
			enc_in.append(pad(e,enc_size)[::-1])
		enc_b = np.swapaxes(np.array(enc_in),0,1)
		dec_b = np.swapaxes(np.array(dec_in),0,1)
		mask_b = [np.array([0. if lid==dec_size-1 or \
			lid<dec_size-1 and dec_in[bid][lid+1]==self.sid['pad'] else 1. \
			for bid in xrange(b_size)]) for lid in xrange(dec_size)]
		return list(enc_b), list(dec_b), mask_b

class Model(object):
	def __init__(self, fw_only, b_size, encv, decv):
		self.fw_only = fw_only
		self.b_size = b_size
		self.encv = encv
		self.decv = decv
		self.n_samp = 512
		self.h_dim = 256
		self.o_dim = decv
		self.n_layer = 3
		self.lr = 0.5
		self.max_grad = 5.
	def build(self):
		self.enc_in = [tf.placeholder(tf.int32,shape=[None],name='encoder{}'.format(i))
						for i in range(config['buckets'][-1][0])]
		self.dec_in = [tf.placeholder(tf.int32,shape=[None],name='decoder{}'.format(i))
						for i in range(config['buckets'][-1][1]+1)]
		self.dec_ma = [tf.placeholder(tf.float32,shape=[None],name='mask{}'.format(i))
						for i in range(config['buckets'][-1][1]+1)]
		self.targets = self.dec_in[1:]
		assert 0 < self.n_samp < self.o_dim
		w = tf.get_variable('proj_w',[self.h_dim,self.o_dim])
		b = tf.get_variable('proj_b',[self.o_dim])
		self.out_proj = (w,b)
		self.loss_func = lambda logits,labels: tf.nn.sampled_softmax_loss(weights=tf.transpose(w), 
																		  biases=b, 
																		  inputs=logits, 
																		  labels=tf.reshape(labels,[-1,1]), 
																		  num_sampled=self.n_samp, 
																		  num_classes=self.o_dim)
		sing_cell = tf.contrib.rnn.GRUCell(self.h_dim)
		self.cell = tf.contrib.rnn.MultiRNNCell([sing_cell for _ in range(self.n_layer)])
		def seq2seq(enc_in, dec_in, do_dec):
			setattr(tf.contrib.rnn.GRUCell,'__deepcopy__',lambda self,_:self)
			setattr(tf.contrib.rnn.MultiRNNCell,'__deepcopy__',lambda self,_:self)
			return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
						enc_in,dec_in,self.cell,
						num_encoder_symbols=self.encv,
						num_decoder_symbols=self.decv,
						embedding_size=self.h_dim,
						output_projection=self.out_proj,
						feed_previous=do_dec)
		self.out,self.loss = tf.contrib.legacy_seq2seq.model_with_buckets(
						self.enc_in,self.dec_in,self.targets,self.dec_ma,config['buckets'], 
						seq2seq=lambda x,y:seq2seq(x,y,self.fw_only),
						softmax_loss_function=self.loss_func)
		if self.fw_only:
			for bid in xrange(len(config['buckets'])):
				self.out[bid] = [tf.matmul(o,self.out_proj[0])+self.out_proj[1] for o in self.out[bid]]
		with tf.variable_scope('training') as scope:
			self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
			if not self.fw_only:
				self.opt = tf.train.GradientDescentOptimizer(self.lr)
				trainables = tf.trainable_variables()
				self.grad_norms,self.train_ops = [],[]
				for bid in xrange(len(config['buckets'])):
					grads,norm = tf.clip_by_global_norm(tf.gradients(self.loss[bid],trainables),self.max_grad)
					self.grad_norms.append(norm)
					self.train_ops.append(self.opt.apply_gradients(zip(grads,trainables),global_step=self.global_step))

class ChatBot(object):
	def __init__(self, data, encv, decv):
		self.data = data
		self.encv = encv
		self.decv = decv
		self.b_size = 64
		self.proc_dir = 'data/cornell/processed'
		self.cpt_dir = 'data/cornell/checkpoints'
		self.out_fn = 'data/cornell/processed/output_convo.txt'
	def check_restore(self, sess, saver):
		ckpt = tf.train.get_checkpoint_state(self.cpt_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
	def run_step(self, sess, enc_in, dec_in, dec_ma, bid, fw_only):
		enc_size,dec_size = config['buckets'][bid]
		input_feed = {}
		for step in xrange(enc_size):
			input_feed[self.model.enc_in[step].name] = enc_in[step]
		for step in xrange(dec_size):
			input_feed[self.model.dec_in[step].name] = dec_in[step]
			input_feed[self.model.dec_ma[step].name] = dec_ma[step]
		input_feed[self.model.dec_in[dec_size].name] = np.zeros([self.model.b_size],dtype=np.int32)
		output_feed = [self.model.loss[bid]] if fw_only else\
					  [self.model.loss[bid],self.model.train_ops[bid],self.model.grad_norms[bid]]
		for step in xrange(dec_size): output_feed.append(self.model.out[bid][step])
		outputs = sess.run(output_feed,input_feed)
		return outputs[0], outputs[1:]
	def eval_test_set(self, sess, te_buc):
		for bid in xrange(len(config['buckets'])):
			if len(te_buc[bid]) == 0: continue
			enc_in,dec_in,dec_ma = self.data.get_batch(te_buc[bid],bid,b_size=self.b_size)
			print bid, self.run_step(sess,enc_in,dec_in,dec_ma,bid,True)[0]
	def train(self):
		def get_buckets():
			tr_buc = self.data.load_data('train_ids.enc','train_ids.dec')
			te_buc = self.data.load_data('test_ids.enc','test_ids.dec')
			tr_buc_sizes = [len(tr_buc[b]) for b in xrange(len(config['buckets']))]
			tr_buc_scale = [sum(tr_buc_sizes[:i+1])/sum(tr_buc_sizes) for i in xrange(len(tr_buc_sizes))]
			return te_buc, tr_buc, tr_buc_scale
		def get_skip_step(it):
			return 30 if it < 100 else 100
		def get_random_bucket(tr_buc_scale):
			rand = random.random()
			return min([i for i in xrange(len(tr_buc_scale)) if tr_buc_scale[i]>rand])
		te_buc, tr_buc, tr_buc_scale = get_buckets()
		self.model = Model(False,self.b_size,self.encv,self.decv)
		self.model.build()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.check_restore(sess,saver)
			it = self.model.global_step.eval()
			total_loss = 0
			while True:
				skip_step = get_skip_step(it)
				bid = get_random_bucket(tr_buc_scale)
				enc_in,dec_in,dec_ma = self.data.get_batch(tr_buc[bid],bid,b_size=self.b_size)
				total_loss += self.run_step(sess,enc_in,dec_in,dec_ma,bid,False)[0]
				if it%skip_step == 0:
					print total_loss/skip_step; total_loss = 0
					saver.save(sess,os.path.join(self.cpt_dir,'chatbot'),global_step=self.model.global_step)
					if it%(10*skip_step) == 0:
						self.eval_test_set(sess,te_buc)
	def chat(self):
		def load_vocab(fn):
			return {w:i for i,w in enumerate(open(os.path.join(\
					self.proc_dir,fn)).read().strip().split('\n'))}
		def get_request():
			return raw_input('> ')
		def find_bucket(length):
			return min([b for b in xrange(len(config['buckets']))\
				if config['buckets'][b][0]>=length])
		def get_response(logits, dec_vocab):
			outputs = [int(np.argmax(logit,axis=1)) for logit in logits]
			if self.data.sid['E'] in outputs:
				outputs = outputs[:outputs.index(self.data.sid['E'])]
			return ' '.join([tf.compat.as_str(dec_vocab[i]) for i in outputs])
		enc_vocab = load_vocab('vocab.enc')
		dec_vocab = {i:w for w,i in load_vocab('vocab.dec').iteritems()}
		self.model = Model(True,1,self.encv,self.decv)
		self.model.build()
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.check_restore(sess,saver)
			fout = open(self.out_fn,'a+')
			max_len = config['buckets'][-1][0]
			while True:
				req = get_request().strip()
				if req == '': break
				fout.write('HUMAN: '+req+'\n')
				token_ids = self.data.sent2id(enc_vocab,str(req))
				assert len(token_ids) <= max_len
				bid = find_bucket(len(token_ids))
				enc_in,dec_in,dec_ma = data.get_batch([(token_ids,[])],bid,b_size=1)
				_,logits = self.run_step(sess,enc_in,dec_in,dec_ma,bid,True)
				resp = get_response(logits,dec_vocab); print resp
				fout.write('BOT: '+resp+'\n')
			fout.close()

if __name__ == '__main__':
	data = DataProc()
	# data.process_data()
	encv,decv = data.process_data()
	bot = ChatBot(data,encv,decv)
	bot.train()
	bot.chat()
