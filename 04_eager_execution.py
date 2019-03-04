# -*- coding: utf-8 -*-

import tensorflow as tf

def test_eager():
	import tensorflow.contrib.eager as tfe
	tfe.enable_eager_execution()
	x = [[2.]]
	print tf.matmul(x,x)
	x = tf.constant([1.,2.,3.])
	for i in x: print i
	square = lambda x:x**2
	grad = tfe.gradients_function(square)
	print grad(3.)
	x = tfe.Variable(2.0)
	loss = lambda y:(y-x**2)**2
	grad = tfe.implicit_gradients(loss)
	print grad(7.)
	
if __name__ == '__main__':
	test_eager()
