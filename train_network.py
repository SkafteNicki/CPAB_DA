#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:25:02 2017

@author: nicki
"""
#%%
import tensorflow as tf
import tflearn
import numpy as np
import time
import math
from utils_func import load_lfw, load_obj

#%%
def create_net(shape, batch_size):
    images_L = tf.placeholder(tf.float32, shape=(shape), name='L')
    images_R = tf.placeholder(tf.float32, shape=(shape), name='R')
    labels =   tf.placeholder(tf.float32, shape=([None,1]), name='gt')
    dropout_f = tf.placeholder("float")

    def build_net(images, dropout_rate):
        net = tflearn.conv_2d(images, 32, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2)
        net = tflearn.local_response_normalization(net)
        net = tflearn.dropout(net, dropout_rate)
        net = tflearn.conv_2d(net, 64, 3, activation='relu')
        net = tflearn.max_pool_2d(net, 2)
        net = tflearn.local_response_normalization(net)
        net = tflearn.dropout(net, dropout_rate)
        net = tflearn.fully_connected(net, 128, activation='tanh')
        net = tflearn.dropout(net, dropout_rate)
        net = tflearn.fully_connected(net, 128, activation='tanh')
        net = tflearn.dropout(net, dropout_rate)
        return net    
    
    def contrastive_loss(y,d):
        tmp= y * tf.square(d)
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
    

    with tf.variable_scope("siemese"):
        model1 = build_net(images_L, dropout_f)
        tf.get_variable_scope().reuse_variables()
        model2 = build_net(images_R, dropout_f)
    
    distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1,model2),2),1,keep_dims=True))
    loss = contrastive_loss(labels,distance)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

    return optimizer, loss, distance
#%%
def compute_accuracy(prediction,labels):
    return labels[prediction.ravel() < 0.5].mean()

#%%
def create_img_pairs(imgs, index, pairs):
    # Reshape into data structure for network
    X_train = np.zeros(shape=(len(pairs['train']), 2, 250, 250, 3), dtype=np.uint8)
    y_train = np.zeros(shape=(len(pairs['train'])), dtype=np.uint8)
    count = 0    
    for n in np.random.permutation(len(pairs['train'])):
        X_train[count,0] = imgs[index[pairs['train'][n][0]][pairs['train'][n][1]-1]]
        X_train[count,1] = imgs[index[pairs['train'][n][2]][pairs['train'][n][3]-1]]
        y_train[count] = 1 if pairs['train'][n][0] == pairs['train'][n][2] else 0
        count += 1        
        
    X_val = np.zeros(shape=(len(pairs['val']), 2, 250, 250, 3), dtype=np.uint8)    
    y_val = np.zeros(shape=(len(pairs['val'])), dtype=np.uint8)
    count = 0
    for n in np.random.permutation(len(pairs['val'])):
        X_val[count,0] = imgs[index[pairs['val'][n][0]][pairs['val'][n][1]-1]]
        X_val[count,1] = imgs[index[pairs['val'][n][2]][pairs['val'][n][3]-1]]
        y_val[count] = 1 if pairs['val'][n][0] == pairs['val'][n][2] else 0
        count += 1
        
    X_test = np.zeros(shape=(10, len(pairs['test'][0]), 2, 250, 250, 3), dtype=np.uint8)
    y_test = np.zeros(shape=(10, len(pairs['test'][0])), dtype=np.uint8)
    for i in range(10):
        count = 0
        for n in np.random.permutation(len(pairs['test'][i])):
            X_test[i,count,0] = imgs[index[pairs['test'][i][n][0]][pairs['test'][i][n][1]-1]]
            X_test[i,count,1] = imgs[index[pairs['test'][i][n][2]][pairs['test'][i][n][3]-1]]   
            y_test[i,count] = 1 if pairs['test'][i][n][0] == pairs['test'][i][n][2] else 0
            count += 1
    return X_train, y_train, X_val, y_val, X_test, y_test 

#%% Main script
if __name__ == '__main__':
    # Load lfw data
    imgs, index, landmarks, pairs = load_lfw()

    # Load estimated cluster parameters    
    _, _, _, Nk, _ = load_obj('cluster_data/cluster_parameters_processed') 
    
    # Create data
    print("Creating data")
    X_train, y_train, X_val, y_val, X_test, y_test = create_img_pairs(imgs, index, pairs)

    # Parameters
    # TODO: Change to argparser
    augment_type = 0
    batch_size = 20
    n_epochs = 300
    
    # Load presampled transformations
    #trans = load_obj('transformations')
    
    def get_batch_no_augmentation(idx, batch_size):
        return (X_train[(idx*batch_size):((idx+1)*batch_size), 0], 
                X_train[(idx*batch_size):((idx+1)*batch_size), 1],
                y_train[(idx*batch_size):((idx+1)*batch_size)])
    def get_batch_normal_augmentation(idx, batch_size):
        return (None, None, None)
    def get_batch_cpab_augmentation(idx, batch_size):
        return (None, None, None)
    def get_batch(augment_type, idx, batch_size):
        return {0: get_batch_no_augmentation(idx, batch_size),
                1: get_batch_normal_augmentation(idx, batch_size),
                2: get_batch_cpab_augmentation(idx, batch_size)
                }[augment_type]
    
    
    # Launch graph
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
#        global_step = tf.Variable(0,trainable=False)
#        images_L = tf.placeholder(tf.float32,shape=([None,784]),name='L')
#        images_R = tf.placeholder(tf.float32,shape=([None,784]),name='R')
#        labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
#        dropout_f = tf.placeholder("float")
#    
#        def mlp(input_,input_dim,output_dim,name="mlp"):
#            with tf.variable_scope(name):
#                w = tf.get_variable('w',[input_dim,output_dim],tf.float32,tf.random_normal_initializer(mean = 0.001,stddev=0.02))
#                return tf.nn.relu(tf.matmul(input_,w))
#        
#        def mlpnet(image,_dropout):
#            l1 = mlp(image,784,128,name='l1')
#            l1 = tf.nn.dropout(l1,_dropout)
#            l2 = mlp(l1,128,128,name='l2')
#            l2 = tf.nn.dropout(l2,_dropout)
#            l3 = mlp(l2,128,128,name='l3')
#            return l3
#        
#        def build_model_mlp(X_,_dropout):
#
#            model = mlpnet(X_,_dropout)
#            return model
#    
#        with tf.variable_scope("siamese") as scope:
#            model1= build_model_mlp(images_L,dropout_f)
#            scope.reuse_variables()
#            model2 = build_model_mlp(images_R,dropout_f)
    
        # Create network
        print("Setting up network")
        optimizer, loss, distance = create_net([None, 250, 250, 3], batch_size)
#    
#        # Training cycle
#        for epoch in range(30):
#            avg_loss = 0.0
#            avg_acc = 0.0
#            total_num_batches = int(X_train.shape[0] / batch_size)
#            start_time = time.time()
#            for i in range(total_num_batches):
#                input1, input2, y = get_batch(augment_type, i)
#                _, loss_value, predict = sess.run([optimizer, loss, distance],
#                                                  feed_dict = {images_L:  input1,
#                                                               images_R:  input2,
#                                                               labels:    y,
#                                                               dropout_r: 0.5})
#                tr_acc = compute_accuracy(predict, y)
#                if math.isnan(tr_acc) and epoch != 0:
#                    print('tr_acc %0.2f' % tr_acc)
#                    avg_loss += loss_value
#                    avg_acc += tr_acc * 100
#                    duration = time.time() - start_time
    