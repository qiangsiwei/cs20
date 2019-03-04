# -*- coding: utf-8 -*-

import os,  scipy.io, scipy.misc, urllib, numpy as np, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from PIL import Image, ImageOps

def resized_image(img_path, img_w, img_h, save=True):
    image = Image.open(img_path)
    image = ImageOps.fit(image,(img_w,img_h),Image.ANTIALIAS)
    out_path = os.path.join(os.path.dirname(img_path),
               'resized_'+os.path.basename(img_path))
    if not os.path.exists(out_path): image.save(out_path)
    return np.expand_dims(np.asarray(image,np.float32),0)

def noise_image(con_image, img_w, img_h, ratio=0.6):
    noise = np.random.uniform(-20,20,(1,img_h,img_w,3)).astype(np.float32)
    return noise*ratio+con_image*(1-ratio)

def save_image(path, image):
    scipy.misc.imsave(path,np.clip(image[0],0,255).astype('uint8'))

class VGG(object):
    def __init__(self, input_img):
        fn = self.download()
        self.vgg_layers = scipy.io.loadmat(fn)['layers']
        self.input_img = input_img
        self.mean = np.array([123.68,116.779,103.939]).reshape((1,1,1,3))
    def download(self):
        lk = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
        fn = 'data/style/imagenet-vgg-verydeep-19.mat'
        if os.path.exists(fn): return fn
        return urllib.urlretrieve(lk,fn)[0]
    def conv2d_relu(self, prev_layer, layer_idx, layer_name):
        def weights(layer_idx, layer_name):
            W = self.vgg_layers[0][layer_idx][0][0][2][0][0]
            b = self.vgg_layers[0][layer_idx][0][0][2][0][1]
            assert self.vgg_layers[0][layer_idx][0][0][0][0] == layer_name
            return W, b.reshape(b.size)
        with tf.variable_scope(layer_name) as scope:
            W,b = weights(layer_idx,layer_name)
            W = tf.constant(W,name='weights')
            b = tf.constant(b,name='bias')
            conv2d = tf.nn.conv2d(prev_layer,filter=W,strides=[1,1,1,1],padding='SAME')
            out = tf.nn.relu(conv2d+b)
        setattr(self,layer_name,out)
    def avgpool(self, prev_layer, layer_name):
        with tf.variable_scope(layer_name):
            out = tf.nn.avg_pool(prev_layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        setattr(self,layer_name,out)
    def load(self):
        self.conv2d_relu(self.input_img,0,'conv1_1')
        self.conv2d_relu(self.conv1_1,2,'conv1_2')
        self.avgpool(self.conv1_2,'avgpool1')
        self.conv2d_relu(self.avgpool1,5,'conv2_1')
        self.conv2d_relu(self.conv2_1,7,'conv2_2')
        self.avgpool(self.conv2_2,'avgpool2')
        self.conv2d_relu(self.avgpool2,10,'conv3_1')
        self.conv2d_relu(self.conv3_1,12,'conv3_2')
        self.conv2d_relu(self.conv3_2,14,'conv3_3')
        self.conv2d_relu(self.conv3_3,16,'conv3_4')
        self.avgpool(self.conv3_4,'avgpool3')
        self.conv2d_relu(self.avgpool3,19,'conv4_1')
        self.conv2d_relu(self.conv4_1,21,'conv4_2')
        self.conv2d_relu(self.conv4_2,23,'conv4_3')
        self.conv2d_relu(self.conv4_3,25,'conv4_4')
        self.avgpool(self.conv4_4,'avgpool4')
        self.conv2d_relu(self.avgpool4,28,'conv5_1')
        self.conv2d_relu(self.conv5_1,30,'conv5_2')
        self.conv2d_relu(self.conv5_2,32,'conv5_3')
        self.conv2d_relu(self.conv5_3,34,'conv5_4')
        self.avgpool(self.conv5_4,'avgpool5')

class StyleTransfer(object):
    def __init__(self, con_img, sty_img, img_w, img_h):
        self.img_w = img_w
        self.img_h = img_h
        self.con_img = resized_image(con_img,img_w,img_h)
        self.sty_img = resized_image(sty_img,img_w,img_h)
        self.ini_img = noise_image(self.con_img,img_w,img_h)
        self.con_layer = 'conv4_2'
        self.sty_layer = ['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1']
        self.alpha = 0.01
        self.sty_w = [0.5,1.0,1.5,3.0,4.0]
        self.lr = 2.0
        self.build_model()
    def build_model(self):
        self.input_img = tf.get_variable('img',shape=([1,self.img_h,self.img_w,3]),
            dtype=tf.float32,initializer=tf.zeros_initializer())
        self.vgg = VGG(self.input_img)
        self.vgg.load()
        self.con_img -= self.vgg.mean
        self.sty_img -= self.vgg.mean
        def con_loss(P):
            return tf.reduce_sum((getattr(self.vgg,self.con_layer)-P)**2)/(4.0*P.size)
        def sty_loss(A):
            def ss_loss(a, b):
                M,N = a.shape[1]*a.shape[2],a.shape[3]
                def gm(x): 
                    x = tf.reshape(x,(M,N))
                    return tf.matmul(tf.transpose(x),x)
                return tf.reduce_sum((gm(b)-gm(a))**2/((2*M*N)**2))
            return sum([self.sty_w[i]*ss_loss(A[i],getattr(self.vgg,self.sty_layer[i])) for i in xrange(len(A))])
        with tf.variable_scope('losses') as scope:
            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.con_img))
                self.con_loss = con_loss(sess.run(getattr(self.vgg,self.con_layer)))
            with tf.Session() as sess:
                sess.run(self.input_img.assign(self.sty_img))
                self.sty_loss = sty_loss(sess.run([getattr(self.vgg,l) for l in self.sty_layer]))
            self.loss = self.alpha*self.con_loss+self.sty_loss
        self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,name='gstep')
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss,global_step=self.gstep)
        with tf.name_scope('summaries'):
            tf.summary.scalar('con loss',self.con_loss)
            tf.summary.scalar('sty loss',self.sty_loss)
            tf.summary.scalar('total loss',self.loss)
            self.summary = tf.summary.merge_all()
    def train(self, n_epochs=300, skip_step=20, dirn='data/style'):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(os.path.join(dirn,'graph'),sess.graph)
            sess.run(self.input_img.assign(self.ini_img))
            step = self.gstep.eval()
            for epoch in range(n_epochs):
                sess.run(self.opt)
                if (epoch+1)%skip_step == 0:
                    gen_img,l,s = sess.run([self.input_img,self.loss,self.summary])
                    save_image(os.path.join(dirn,'output','{}.png'.format(epoch)),gen_img+self.vgg.mean)
                    writer.add_summary(s,global_step=epoch); print epoch,l
                    
if __name__ == '__main__':
    StyleTransfer('data/style/deadpool.jpg','data/style/guernica.jpg',333,250).train()
