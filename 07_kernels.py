# -*- coding: utf-8 -*-

import os, numpy as np, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from matplotlib import gridspec as gridspec
from matplotlib import pyplot as plt

a = np.zeros([3, 3, 1, 1])
a[1, 1, :, :] = 1.0
a[0, 1, :, :] = 1.0
a[1, 0, :, :] = 1.0
a[2, 1, :, :] = 1.0
a[1, 2, :, :] = 1.0
a[0, 0, :, :] = 1.0
a[0, 2, :, :] = 1.0
a[2, 0, :, :] = 1.0
a[2, 2, :, :] = 1.0
BLUR_FILTER = tf.constant(a,dtype=tf.float32)
a = np.zeros([3, 3, 1, 1])
a[1, 1, :, :] = 5
a[0, 1, :, :] = -1
a[1, 0, :, :] = -1
a[2, 1, :, :] = -1
a[1, 2, :, :] = -1
SHARPEN_FILTER = tf.constant(a,dtype=tf.float32)
a = np.zeros([3, 3, 1, 1])
a[0, 1, :, :] = -1
a[1, 0, :, :] = -1
a[1, 2, :, :] = -1
a[2, 1, :, :] = -1
a[1, 1, :, :] = 4
EDGE_FILTER = tf.constant(a,dtype=tf.float32)
a = np.zeros([3, 3, 1, 1])
a[0, :, :, :] = 1
a[0, 1, :, :] = 2
a[2, :, :, :] = -1
a[2, 1, :, :] = -2
TOP_SOBEL = tf.constant(a,dtype=tf.float32)
a = np.zeros([3, 3, 1, 1])
a[0, 0, :, :] = -2
a[0, 1, :, :] = -1 
a[1, 0, :, :] = -1
a[1, 1, :, :] = 1
a[1, 2, :, :] = 1
a[2, 1, :, :] = 1
a[2, 2, :, :] = 2
EMBOSS_FILTER = tf.constant(a,dtype=tf.float32)

def read_image(fn):
    return tf.cast(tf.image.decode_image(tf.read_file(fn)),tf.float32)/256.0

def convolve(image, kernels, strides=[1,3,3,1], padding='SAME'):
    images = [image[0]]
    for i,kernel in enumerate(kernels):
        filtered_image = tf.nn.conv2d(image,kernel,strides=strides,padding=padding)[0]
        if i == 2: filtered_image = tf.minimum(tf.nn.relu(filtered_image),255)
        images.append(filtered_image)
    return images

def show_images(images):
    gs = gridspec.GridSpec(1,len(images))
    for i,image in enumerate(images):
        plt.subplot(gs[0,i])
        plt.imshow(image.reshape(image.shape[0],image.shape[1]),cmap='gray')
        plt.axis('off')
    plt.show()

def main():
    kernels_list = [BLUR_FILTER,
                    SHARPEN_FILTER,
                    EDGE_FILTER,
                    TOP_SOBEL,
                    EMBOSS_FILTER]
    kernels_list = kernels_list[1:]
    image = read_image('data/friday.jpg')
    image = tf.image.rgb_to_grayscale(image)
    image = tf.expand_dims(image,0)
    images = convolve(image,kernels_list)
    with tf.Session() as sess:
        images = sess.run(images)
    show_images(images)

if __name__ == '__main__':
    main()
