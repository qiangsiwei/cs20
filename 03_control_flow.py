# -*- coding: utf-8 -*-

import numpy as np, tensorflow as tf

def test_group():
	x = tf.Variable(1,name='x')
	add,mul = tf.add(x,2),tf.multiply(x,2)
	group = tf.group(add,mul)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run([add,mul])

def test_count_up_to():
	x = tf.Variable(1,name='x')
	l = tf.count_up_to(x,4)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(5): print sess.run(l)

def test_cond():
	x = tf.Variable(1,name='x')
	y = tf.Variable(2,name='y')
	s = tf.cond(tf.greater(x,y),lambda:x/y,lambda:y/x)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(s)

def test_case():
	def f(k): return tf.constant(100)*k
	i = tf.constant(2)
	c0 = tf.equal(i,0),lambda:f(0)
	c1 = tf.equal(i,1),lambda:f(1)
	c2 = tf.equal(i,2),lambda:f(2)
	r = tf.case(pred_fn_pairs=[c0,c1,c2],default=lambda:tf.constant(999))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(r)

def test_while():
	i = tf.constant(0)
	r = tf.while_loop(cond=lambda i:tf.less(i,10),body=lambda i:tf.add(i,1),loop_vars=[i])
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(r)

def test_where():
	a1 = np.array([[1,0,0],[0,1,1]])
	a2 = np.array([[1,2,3],[4,5,6]])
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run(tf.where(tf.equal(a1,0),a2,-a2))

def test_debug():
	x = tf.constant(2.)
	a = tf.Assert(tf.less_equal(tf.reduce_max(x),9),[x])
	p = tf.Print(x,[x],message='debug:')
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print sess.run([tf.is_finite(x),tf.is_inf(x),tf.is_nan(x)])
		sess.run([a,p])

def test_identity():
	x = tf.Variable(1,name='x')
	x_ = tf.assign_add(x,1,name='x_')
	with tf.control_dependencies([x_]):
		y, z = x, tf.identity(x,name='z') 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(5): print sess.run(z) # y
	
if __name__ == '__main__':
	# test_group()
	# test_count_up_to()
	# test_cond()
	# test_case()
	# test_while()
	# test_where()
	# test_debug()
	test_identity()
