# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

def test1():
	dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))
	# dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5,2)))
	# dataset = tf.data.Dataset.from_tensor_slices({
	# 	'a':np.array([1.0,2.0,3.0,4.0,5.0]),
	# 	'b':np.random.uniform(size=(5,2))})
	# dataset = dataset.map(lambda x:x+1)
	# dataset = dataset.shuffle(buffer_size=100)
	# dataset = dataset.repeat(5).batch(2)
	dataset = dataset.repeat()
	iterator = dataset.make_one_shot_iterator()
	o = iterator.get_next()
	with tf.Session() as sess:
		for i in xrange(5): print sess.run(o)

def test2():
	import tensorflow.contrib.eager as tfe
	tfe.enable_eager_execution()
	dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))
	for o in tfe.Iterator(dataset): print o

def test3(): # initializable iterator
	x = tf.placeholder(tf.int64,shape=[])
	dataset = tf.data.Dataset.range(x)
	iterator = dataset.make_initializable_iterator()
	o = iterator.get_next()
	with tf.Session() as sess:
		sess.run(iterator.initializer,feed_dict={x:10})
		for i in xrange(5): print sess.run(o)

def test4(): # reinitializable iterator
	tr = tf.data.Dataset.range(100)
	te = tf.data.Dataset.range(50)
	iterator = tf.data.Iterator.from_structure(tr.output_types,tr.output_shapes)
	o = iterator.get_next()
	tr_init_op = iterator.make_initializer(tr)
	te_init_op = iterator.make_initializer(te)
	with tf.Session() as sess:
		sess.run(tr_init_op)
		for _ in range(10): print sess.run(o)
		sess.run(te_init_op)
		for _ in range(10): print sess.run(o)

def test5(): # feedable iterator
	tr = tf.data.Dataset.range(100)
	te = tf.data.Dataset.range(50)
	handle = tf.placeholder(tf.string,shape=[])
	iterator = tf.data.Iterator.from_string_handle(handle,tr.output_types,tr.output_shapes)
	o = iterator.get_next()
	tr_iterator = tr.make_one_shot_iterator()
	te_iterator = tr.make_initializable_iterator()
	with tf.Session() as sess:
		tr_handle = sess.run(tr_iterator.string_handle())
		te_handle = sess.run(te_iterator.string_handle())
		sess.run(te_iterator.initializer)
		for _ in range(10): print sess.run(o,feed_dict={handle:tr_handle})
		for _ in range(10): print sess.run(o,feed_dict={handle:te_handle})

def test6(): # TFRecordDataset
	from tensorflow.examples.tutorials.mnist import input_data
	def _bytes_feature(value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	def _int64_feature(value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
	mnist = input_data.read_data_sets('data/mnist',dtype=tf.uint8,reshape=True)
	images,labels = mnist.train.images,mnist.train.labels
	writer = tf.python_io.TFRecordWriter('data/mnist.tfrecords')
	for i in xrange(mnist.train.num_examples):
		example = tf.train.Example(features=tf.train.Features(feature={
					'image':_bytes_feature(images[i].tostring()),
					'label':_int64_feature(int(labels[i]))}))
		writer.write(example.SerializeToString())
	writer.close()

def test7(): # TFRecordDataset
	def parse(example_proto):
		features = {'image':tf.FixedLenFeature((),tf.string,default_value=''),
					'label':tf.FixedLenFeature((),tf.int64,default_value=0)}
		parsed_features = tf.parse_single_example(example_proto,features)
		return parsed_features['image'],parsed_features['label']
	dataset = tf.data.TFRecordDataset(['data/mnist/mnist.tfrecords']).map(parse)
	iterator = dataset.make_one_shot_iterator()
	o = iterator.get_next()
	with tf.Session() as sess:
		for i in xrange(5): print sess.run(o)

def test8(): # TextLineDataset
	decode = lambda x:x
	dataset = tf.data.TextLineDataset('data/heart.csv').map(decode)
	iterator = dataset.make_one_shot_iterator()
	o = iterator.get_next()
	with tf.Session() as sess:
		for i in xrange(5): print sess.run(o)

if __name__ == '__main__':
	# test1()
	# test2()
	# test3()
	# test4()
	# test5()
	# test6()
	# test7()
	test8()
