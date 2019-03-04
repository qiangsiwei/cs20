# -*- coding: utf-8 -*-

import os, struct, numpy as np, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def get_mnist(b_size, dirn='data/mnist'):
	def parse_data(dataset, flatten):
		assert dataset in ('train','t10k')
		with open(os.path.join(dirn,dataset+'-labels-idx1-ubyte'),'rb') as fin:
			_,num = struct.unpack('>II',fin.read(8))
			labels = np.zeros((num, 10))
			labels[np.arange(num),np.fromfile(fin,dtype=np.int8)] = 1
		with open(os.path.join(dirn,dataset+'-images-idx3-ubyte'),'rb') as fin:
			_,num,rows,cols = struct.unpack('>IIII',fin.read(16))
			images = np.fromfile(fin,dtype=np.uint8).reshape(num,rows,cols)
			images = images.astype(np.float32)/255.0
			if flatten: images = images.reshape([num,-1])
		return images, labels
	def read_mnist(flatten, tr_num=55000):
		images,labels = parse_data('train',flatten)
		inds = np.random.permutation(labels.shape[0])
		tr_idx,va_idx = inds[:tr_num],inds[tr_num:]
		tr_images,tr_labels = images[tr_idx,:],labels[tr_idx,:]
		va_images,va_labels = images[va_idx,:],labels[va_idx,:]
		test = parse_data('t10k',flatten)
		return (tr_images,tr_labels),(va_images,va_labels),test
	tr,_,te = read_mnist(flatten=False)
	tr_data = tf.data.Dataset.from_tensor_slices(tr).shuffle(10000).batch(b_size)
	te_data = tf.data.Dataset.from_tensor_slices(te).batch(b_size)
	return tr_data, te_data

class ConvNet(object):
	def __init__(self):
		self.n_class, self.keep, self.b_size, self.lr = 10, tf.constant(0.75), 128, 0.001
		self.build_model()
		self.training = False
	def build_model(self):
		def build1():
			def conv_relu(inputs, filters, k_size, stride, padding, scope_name):
				with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope: 
					kernel = tf.get_variable('kernel',[k_size,k_size,inputs.shape[-1],filters],
						initializer=tf.truncated_normal_initializer())
					biases = tf.get_variable('biases',[filters],
						initializer=tf.random_normal_initializer())
					conv = tf.nn.conv2d(inputs,kernel,strides=[1,stride,stride,1],padding=padding)
					return tf.nn.relu(conv+biases,name=scope.name)
			def maxpool(inputs, ksize, stride, padding='VALID', scope_name='pool'):
				with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
					return tf.nn.max_pool(inputs,ksize=[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)
			def fully_connected(inputs, out_dim, scope_name='fc'):
				with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
					w = tf.get_variable('weights',[inputs.shape[-1],out_dim],
						initializer=tf.truncated_normal_initializer())
					b = tf.get_variable('biases',[out_dim],
						initializer=tf.constant_initializer(0.0))
				return tf.matmul(inputs,w)+b
			conv1 = conv_relu(inputs=self.img,filters=32,k_size=5,stride=1,padding='SAME',scope_name='conv1')
			pool1 = maxpool(conv1,2,2,'VALID','pool1')
			conv2 = conv_relu(inputs=pool1,filters=64,k_size=5,stride=1,padding='SAME',scope_name='conv2')
			pool2 = maxpool(conv2,2,2,'VALID','pool2')
			pool2 = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
			fc = fully_connected(pool2,1024,'fc')
			dropout = tf.nn.dropout(tf.nn.relu(fc),self.keep,name='dropout')
			self.logits = fully_connected(dropout,self.n_class,'logits')
		def build2():
			conv1 = tf.layers.conv2d(inputs=self.img,filters=32,kernel_size=[5,5],padding='SAME',activation=tf.nn.relu,name='conv1')
			pool1 = tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2,name='pool1')
			conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[5, 5],padding='SAME',activation=tf.nn.relu,name='conv2')
			pool2 = tf.layers.max_pooling2d(inputs=conv2,pool_size=[2, 2],strides=2,name='pool2')
			pool2 = tf.reshape(pool2,[-1,pool2.shape[1]*pool2.shape[2]*pool2.shape[3]])
			fc = tf.layers.dense(pool2,1024,activation=tf.nn.relu,name='fc')
			dropout = tf.layers.dropout(fc,1-self.keep,training=self.training,name='dropout')
			self.logits = tf.layers.dense(dropout,self.n_class,name='logits')
		tr_data,te_data = get_mnist(self.b_size)
		iterator = tf.data.Iterator.from_structure(tr_data.output_types,tr_data.output_shapes)
		img, self.label = iterator.get_next()
		self.img = tf.reshape(img,shape=[-1,28,28,1])
		self.tr_init = iterator.make_initializer(tr_data)
		self.te_init = iterator.make_initializer(te_data)
		build1()
		# build2()
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label,logits=self.logits),name='loss')
		self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
		self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.gstep)
		preds = tf.nn.softmax(self.logits)
		correct_preds = tf.equal(tf.argmax(preds,1),tf.argmax(self.label,1))
		self.accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
		tf.summary.scalar('loss',self.loss)
		tf.summary.scalar('accuracy',self.accuracy)
		tf.summary.histogram('histogram loss',self.loss)
		self.summary = tf.summary.merge_all()
	def train(self, n_epochs=30, skip_step=20, dirn='data/convnet'):
		saver = tf.train.Saver()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			ckpt = tf.train.get_checkpoint_state(os.path.join(dirn,'checkpoint'))
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
			loss_, batch_, accu_ = 0, 0, 0
			writer = tf.summary.FileWriter(os.path.join(dirn,'graph'),sess.graph)
			step = self.gstep.eval()
			for epoch in xrange(n_epochs):
				sess.run(self.tr_init); self.training = True
				try:
					while True:
						_,l,s = sess.run([self.opt,self.loss,self.summary])
						writer.add_summary(s,global_step=step)
						# if (step+1)%skip_step == 0:
						# 	print loss_; loss_ = 0
						loss_ += l; batch_ += 1; step += 1
				except tf.errors.OutOfRangeError: pass
				saver.save(sess,os.path.join(dirn,'checkpoint'),step)
				print 'avg loss', loss_/batch_; loss_ = batch_ = 0
				sess.run(self.te_init); self.training = False
				try:
					while True:
						a,s = sess.run([self.accuracy,self.summary])
						writer.add_summary(s,global_step=step)
						accu_ += a
				except tf.errors.OutOfRangeError: pass
				print 'avg accu', 1.*accu_/10000; accu_ = 0
			writer.close()

if __name__ == '__main__':
	cn = ConvNet()
	cn.train()
