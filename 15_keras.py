# -*- coding: utf-8 -*-

def toy_video_QA():
	import tensorflow as tf
	video = tf.keras.layers.Input(shape=(None,150,150,3))
	cnn = tf.keras.applications.InceptionV3(weights='imagenet',include_top=False,pooling='avg')
	cnn.trainable = False
	encoded_frames = tf.keras.layers.TimeDistributed(cnn)(video)
	encoded_vid = tf.keras.layers.LSTM(256)(encoded_frames)
	question = tf.keras.layers.Input(shape=[100],dtype='int32')
	x = tf.keras.layers.Embedding(10000,256,mask_zero=True)(question)
	encoded_q = tf.keras.layers.LSTM(128)(x)
	x = tf.keras.layers.concatenate([encoded_vid,encoded_q])
	x = tf.keras.layers.Dense(128,activation=tf.nn.relu)(x)
	outputs = tf.keras.layers.Dense(1000)(x)
	model = tf.keras.models.Model([video,question],outputs)
	model.compile(optimizer='adam',loss='mean_absolute_percentage_error')

def multi_GPU():
	import numpy as np, tensorflow as tf
	from keras.applications import Xception
	from keras.utils import multi_gpu_model
	height, width = 224, 224
	num_classes = 1000
	num_samples = 1000
	with tf.device('/cpu:0'):
		model = Xception(weights=None,input_shape=(height,width,3),classes=num_classes)
	parallel_model = multi_gpu_model(model,gpus=4)
	parallel_model.compile(loss='categorical_crossentropy',optimizer='rmsprop')
	x = np.random.random((num_samples,height,width,3))
	y = np.random.random((num_samples,num_classes))
	parallel_model.fit(x,y,epochs=20,batch_size=256)
	model.save('my_model.h5')

def eager_exec():
	import keras
	from keras import layers
	import tensorflow.contrib.eager as tfe
	tfe.enable_eager_execution()
	inputs = keras.Input(shape=(10,))
	x = layers.Dense(20,activation='relu')(x)
	x = layers.Dense(20,activation='relu')(x)
	outputs = layers.Dense(10,activation='softmax')(x)
	model = keras.Model(inputs,outputs)

if __name__ == '__main__':
    # toy_video_QA()
    # multi_GPU()
    eager_exec()
