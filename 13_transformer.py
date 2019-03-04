# -*- coding: utf-8 -*-

import tensorflow as tf
from keras.models import Model
from keras.layers import *
from keras.engine.topology import Layer
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import backend as K

def test_keras(): # no mask
    class Position_Embedding(Layer):
        def __init__(self, **kwargs):
            super(Position_Embedding,self).__init__(**kwargs)
        def call(self, x):
            self.size = int(x.shape[-1])
            pij = K.dot(K.expand_dims(K.cumsum(K.ones_like(x[:,:,0]),1)-1,2),\
                K.expand_dims(1./K.pow(10000.,2*K.arange(self.size/2,dtype='float32')/self.size),0))
            return K.concatenate([K.cos(pij),K.sin(pij)],2)+x
        def compute_output_shape(self, input_shape):
            return input_shape
    class Attention(Layer):
        def __init__(self, nb_head, size_per_head, **kwargs):
            self.nb_head = nb_head
            self.size_per_head = size_per_head
            self.out_dim = nb_head*size_per_head
            super(Attention,self).__init__(**kwargs)
        def build(self, input_shape):
            self.WQ = self.add_weight(name='WQ',shape=(input_shape[0][-1],self.out_dim),initializer='glorot_uniform',trainable=True)
            self.WK = self.add_weight(name='WK',shape=(input_shape[1][-1],self.out_dim),initializer='glorot_uniform',trainable=True)
            self.WV = self.add_weight(name='WV',shape=(input_shape[2][-1],self.out_dim),initializer='glorot_uniform',trainable=True)
            super(Attention,self).build(input_shape)        
        def call(self, x):
            Q_seq,K_seq,V_seq = x
            Q_seq = K.dot(Q_seq,self.WQ)
            Q_seq = K.reshape(Q_seq,(-1,K.shape(Q_seq)[1],self.nb_head,self.size_per_head))
            Q_seq = K.permute_dimensions(Q_seq,(0,2,1,3))
            K_seq = K.dot(K_seq,self.WK)
            K_seq = K.reshape(K_seq,(-1,K.shape(K_seq)[1],self.nb_head,self.size_per_head))
            K_seq = K.permute_dimensions(K_seq,(0,2,1,3))
            V_seq = K.dot(V_seq,self.WV)
            V_seq = K.reshape(V_seq,(-1,K.shape(V_seq)[1],self.nb_head,self.size_per_head))
            V_seq = K.permute_dimensions(V_seq,(0,2,1,3))
            A = K.batch_dot(Q_seq,K_seq,axes=[3,3])/self.size_per_head**0.5
            A = K.permute_dimensions(A,(0,3,2,1))
            A = K.permute_dimensions(A,(0,3,2,1))    
            A = K.softmax(A)
            O_seq = K.batch_dot(A,V_seq,axes=[3,2])
            O_seq = K.permute_dimensions(O_seq,(0,2,1,3))
            O_seq = K.reshape(O_seq,(-1,K.shape(O_seq)[1],self.out_dim))
            return O_seq
        def compute_output_shape(self, input_shape):
            return input_shape[0][0],input_shape[0][1],self.out_dim
    x = Input(shape=(None,),dtype='int32')
    embeddings = Embedding(v_size,128)(x)
    O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    y = Dense(1,activation='sigmoid')(O_seq)
    model = Model(inputs=x,outputs=y)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(x_tr,y_tr,batch_size=b_size,epochs=5,validation_data=(x_te, y_te))

def test_tflow(): # no mask
    class Transformer():
        def __init__(self, x_len):
            self.x_len = x_len
            self.e_dim = 128
            self.n_class = 2
            self.dropout = .2
            self.decay_step = 200
            self.decay_rate = 1.
            self.clip_grad = 5.
            self.b_size = 32
            self.lr = .001
        def build_model(self, v_size):
            self.initializer = tf.random_normal_initializer(stddev=0.1)
            self.x = tf.placeholder(tf.int32,[None,self.x_len],name='x')
            self.y = tf.placeholder(tf.int32,[None],name='y')
            self.embedding = tf.get_variable('embedding',shape=[v_size,self.e_dim],initializer=self.initializer)
            self.embed = tf.nn.embedding_lookup(self.embedding,self.x)
            def pos_embed(x):
                pij = tf.matmul(tf.expand_dims(tf.range(tf.cast(self.x_len,tf.float32),dtype=tf.float32),1),\
                    tf.expand_dims(1./tf.pow(10000.,2*tf.range(self.e_dim/2,dtype=tf.float32)/self.e_dim),0))
                pij = tf.concat([tf.cos(pij),tf.sin(pij)],1)
                return tf.expand_dims(pij,0)+x
            self.embed = pos_embed(self.embed)
            def Dense(x, size_o):
                W = tf.Variable(tf.random_uniform([self.e_dim,size_o],-0.05,0.05))
                out = tf.matmul(tf.reshape(x,(-1,self.e_dim)),W)
                return tf.reshape(out,tf.concat([tf.shape(x)[:-1],[size_o]],0))
            def Attention(Q, K, V, nb_head=8, size_per_head=16):
                Q = Dense(Q,nb_head*size_per_head)
                Q = tf.reshape(Q,(-1,tf.shape(Q)[1],nb_head,size_per_head))
                Q = tf.transpose(Q,[0,2,1,3])
                K = Dense(K,nb_head*size_per_head)
                K = tf.reshape(K,(-1,tf.shape(K)[1],nb_head,size_per_head))
                K = tf.transpose(K,[0,2,1,3])
                V = Dense(V,nb_head*size_per_head)
                V = tf.reshape(V,(-1,tf.shape(V)[1],nb_head,size_per_head))
                V = tf.transpose(V,[0,2,1,3])
                A = tf.matmul(Q,K,transpose_b=True)/tf.sqrt(float(size_per_head))
                A = tf.transpose(A,[0,3,2,1])
                A = tf.transpose(A,[0,3,2,1])
                A = tf.nn.softmax(A)
                O = tf.matmul(A,V)
                O = tf.transpose(O,[0,2,1,3])
                O = tf.reshape(O,(-1,tf.shape(O)[1],nb_head*size_per_head))
                return O
            attn = Attention(self.embed,self.embed,self.embed)
            h = tf.nn.dropout(tf.reduce_mean(attn,axis=1),keep_prob=1-self.dropout)
            W = tf.get_variable('W',shape=[self.e_dim,self.n_class],initializer=self.initializer)
            b = tf.get_variable('b',shape=[self.n_class])
            self.logits = tf.matmul(h,W)+b
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,logits=self.logits))
        def build_train_op(self):  
            self.global_step = tf.Variable(0,trainable=False,name='gs')
            lr = tf.train.exponential_decay(self.lr,self.global_step,self.decay_step,self.decay_rate,staircase=True)
            opt = tf.train.AdamOptimizer(lr)
            g,v = zip(*opt.compute_gradients(self.loss))
            g,_ = tf.clip_by_global_norm(g,self.clip_grad)
            up_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(up_ops):
                self.train_op = opt.apply_gradients(zip(g,v))
            self.pred = tf.argmax(self.logits,1,name='pred')
            self.accu = tf.reduce_mean(tf.cast(tf.equal(tf.cast(self.pred,tf.int32),self.y),tf.float32),name='accu')
        def train(self, x_tr, y_tr, x_te, y_te, v_size, n_epoch=5):
            self.build_model(v_size)
            self.build_train_op()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                n_batch = len(x_tr)/self.b_size
                for epoch in xrange(n_epoch):
                    loss = 0
                    for i in xrange(n_batch):
                        loss += sess.run([self.loss,self.train_op],{\
                            self.x:x_tr[i*self.b_size:(i+1)*self.b_size],\
                            self.y:y_tr[i*self.b_size:(i+1)*self.b_size]})[0]
                    print epoch,'loss:',loss; loss = 0
                    print epoch,'accu:',np.array([sess.run([self.accu],{\
                        self.x:x_te[i*self.b_size:(i+1)*self.b_size],\
                        self.y:y_te[i*self.b_size:(i+1)*self.b_size]})[0]\
                            for i in xrange(len(x_te)/self.b_size)]).mean()
    clf = Transformer(maxlen)
    clf.train(x_tr,y_tr,x_te,y_te,v_size)

if __name__ == '__main__':
    maxlen = 80
    b_size = 32
    v_size = 20000
    (x_tr,y_tr),(x_te,y_te) = imdb.load_data(num_words=v_size)
    x_tr = sequence.pad_sequences(x_tr,maxlen=maxlen)
    x_te = sequence.pad_sequences(x_te,maxlen=maxlen)
    test_keras()
    test_tflow()
