# -*- coding: utf-8 -*-

import pandas as pd, tensorflow as tf

class TestEstimator(object):
	
	def __init__(self):
		self.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
		self.species = ['Setosa','Versicolor','Virginica']
		
	def load_data(self, y_name='Species'):
		def maybe_download():
			TR_URL = 'http://download.tensorflow.org/data/iris_training.csv'
			TE_URL = 'http://download.tensorflow.org/data/iris_test.csv'
			tr_path = tf.keras.utils.get_file(TR_URL.split('/')[-1],TR_URL)
			te_path = tf.keras.utils.get_file(TE_URL.split('/')[-1],TE_URL)
			return tr_path,te_path
		tr_path,te_path = maybe_download()
		tr = pd.read_csv(tr_path,names=self.columns,header=0)
		tr_x,tr_y = tr,tr.pop(y_name)
		te = pd.read_csv(te_path,names=self.columns,header=0)
		te_x,te_y = te,te.pop(y_name)
		return (tr_x,tr_y),(te_x,te_y)

	def build_premade(self, columns):
		return tf.estimator.DNNClassifier(
			feature_columns=columns,
			hidden_units=[10,10],
			n_classes=3)

	def build_custom(self, columns):
		def my_model(features, labels, mode, params):
			net = tf.feature_column.input_layer(features,params['feature_columns'])
			for units in params['hidden_units']:
				net = tf.layers.dense(net,units=units,activation=tf.nn.relu)
			logits = tf.layers.dense(net,params['n_classes'],activation=None)
			output = tf.argmax(logits,1)
			if mode == tf.estimator.ModeKeys.PREDICT:
				return tf.estimator.EstimatorSpec(mode,predictions={
					'class_ids'     :output[:,tf.newaxis],
					'probabilities' :tf.nn.softmax(logits)})
			loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
			accu = tf.metrics.accuracy(labels=labels,predictions=output,name='acc_op')
			metrics = {'accuracy':accu}
			tf.summary.scalar('accuracy',accu[1])
			if mode == tf.estimator.ModeKeys.EVAL:
			    return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=metrics)
			assert mode == tf.estimator.ModeKeys.TRAIN
			optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
			train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
			return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
		return tf.estimator.Estimator(model_fn=my_model,params={
				'feature_columns':columns,
				'hidden_units':[10,10],
				'n_classes':3})

	def tr_input_fn(self, x, y, b_size):
		dataset = tf.data.Dataset.from_tensor_slices((dict(x),y))
		return dataset.shuffle(1000).repeat().batch(b_size)

	def ev_input_fn(self, x, y, b_size):
		x = dict(x) if y is None else (dict(x),y)   
		return tf.data.Dataset.from_tensor_slices(x).batch(b_size)

	def train(self):
		(tr_x,tr_y),(te_x,te_y) = self.load_data()
		columns = [tf.feature_column.numeric_column(key=key) for key in tr_x.keys()]
		# self.clf = self.build_premade(columns)
		self.clf = self.build_custom(columns)
		self.clf.train(input_fn=lambda:self.tr_input_fn(tr_x,tr_y,100),steps=1000) 
		print self.clf.evaluate(input_fn=lambda:self.ev_input_fn(te_x,te_y,100))

	def test(self, x, y):
		pred = self.clf.predict(input_fn=lambda:self.ev_input_fn(x,None,100))
		for pdict,exp in zip(pred,y):
			i = pdict['class_ids'][0]
			p = pdict['probabilities'][i]
			print self.species[i],exp,p

if __name__ == '__main__':
	x = {'SepalLength' :[5.1,5.9,6.9],
		 'SepalWidth'  :[3.3,3.0,3.1],
		 'PetalLength' :[1.7,4.2,5.4],
		 'PetalWidth'  :[0.5,1.5,2.1]}
	y = ['Setosa','Versicolor','Virginica']
	est = TestEstimator()
	est.train()
	est.test(x,y)
