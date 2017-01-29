#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:25:02 2017

@author: nicki
"""
import tensorflow as tf
import tflearn


images_L = tf.placeholder(tf.float32,shape=([None,784]),name='L')
images_R = tf.placeholder(tf.float32,shape=([None,784]),name='R')
labels = tf.placeholder(tf.float32,shape=([None,1]),name='gt')
dropout_f = tf.placeholder("float")


def build_net(images, dropout_rate, augmentation=None):
    if augmentation:
        net = tflearn.input_data(shape=[None, 32, 32, 3])
    else:
        net = tflearn.input_data(shape=[None, 32, 32, 3],
                             data_augmentation=augmentation)
        
    net = tflearn.conv_2d(net, 32, 3, activation='relu')
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
    

with tf.variable_scope("siemese") as scope:
    model1 = build_net(images_L, dropout_f)
    scope.reuse_variables()
    model2 = build_net(images_R, dropout_f)
    
distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(model1,model2),2),1,keep_dims=True))
loss = contrastive_loss(labels,distance)

t_vars = tf.trainable_variables()
d_vars = [var for var in t_vars if 'l' in var.name]
batch = tf.Variable(0)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)

# Launch graph
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    # Training cycle
    for epoch in range(30):
        avg_loss = 0.0
        avg_acc = 0.0
        total_num_batches = int(X_train.shape[0] / batch_size)
        start_time = time.time()
        for i in range(total_num_batches):
            input1, input2, y = next_batch(i)
            _, loss_value, predict = sess.run([optimizer, loss, distance],
                                              feed_dict = {images_L: input1,
                                                           images_R: input2,
                                                           labels:   y,
                                                           dropout_r:0.5})
            tr_acc = compute_accuracy(predict, y)
            if math.isnan(tr_acc) and epoch != 0:
                print('tr_acc %0.2f' % tr_acc)
            avg_loss += loss_value
            avg_acc += tr_acc * 100
        duration = time.time() - start_time
    

imgaug = tflearn.ImageAugmentation()
imgaug.add_random_rotation(20)
imgaug.add_random_crop((20,20))

