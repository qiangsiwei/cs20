# -*- coding: utf-8 -*-

import os, xlrd, numpy as np, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt

def test_placeholder(n_epochs=30):
	rows = xlrd.open_workbook('data/fire_theft.xls').sheet_by_index(0)
	data = np.asarray([rows.row_values(i) for i in xrange(1,rows.nrows)])
	n_samples = rows.nrows-1
	X = tf.placeholder(tf.float32,name='X')
	Y = tf.placeholder(tf.float32,name='Y')
	w = tf.Variable(0.0,name='w')
	b = tf.Variable(0.0,name='b')
	Y_ = X*w+b
	def huber_loss(Y, Y_, dt=1.0):
		r = tf.abs(Y-Y_)
		return tf.cond(r<dt,lambda:0.5*tf.square(r),lambda:dt*r-0.5*tf.square(dt))
	loss = huber_loss(Y,Y_)
	opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('./graphs/lr',sess.graph)
		for i in xrange(n_epochs):
			_loss = 0
			for x,y in data:
				_loss += sess.run([opt,loss],feed_dict={X:x,Y:y})[1]
			print 'Epoch {0}: {1}'.format(i,_loss/n_samples)
		writer.close()
		w,b = sess.run([w,b])
	X,Y = data.T[0],data.T[1]
	plt.plot(X,Y,'bo',label='real')
	plt.plot(X,X*w+b,'r',label='pred')
	plt.legend(); plt.show()

def test_dataset(n_epochs=30, b_size=128, lr=0.01, c=0):
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('data/mnist',one_hot=True)
	X = tf.placeholder(tf.float32,[b_size,784],name='X')
	Y = tf.placeholder(tf.int32,[b_size,10],name='Y')
	w = tf.Variable(tf.random_normal(shape=[784,10],stddev=0.01),name='w')
	b = tf.Variable(tf.zeros([1,10]),name='b')
	Y_ = tf.matmul(X,w)+b
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_,labels=Y))
	opt = tf.train.AdamOptimizer(lr).minimize(loss)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		writer = tf.summary.FileWriter('./graphs/lr',sess.graph) 
		for i in xrange(n_epochs):
			_loss = 0
			for _ in xrange(int(mnist.train.num_examples/b_size)):
				x,y = mnist.train.next_batch(b_size)
				_loss += sess.run([opt,loss],feed_dict={X:x,Y:y})[1]
			print 'Epoch {0}: {1}'.format(i,_loss)
		writer.close()
		pred = tf.nn.softmax(Y_)
		accu = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(pred,1),tf.argmax(Y,1)),tf.float32))
		for i in xrange(int(mnist.test.num_examples/b_size)):
			x,y = mnist.test.next_batch(b_size)
			c += sess.run(accu,feed_dict={X:x,Y:y})
		print 'Accuracy {0}'.format(c/mnist.test.num_examples)

if __name__ == '__main__':
	# test_placeholder()
	test_dataset()
