# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:35:07 2017

@author: Nicki
"""
#%%
from sklearn.datasets import fetch_mldata
import numpy as np
#import matplotlib.pyplot as plt
import pickle as pkl
import argparse
import tflearn as tfl
from datetime import datetime
import time
import os
#%% Function to check if a given file exist
def f_exist(name):
    return os.path.isfile(name)

#%% Function for saving to a pkl file
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
    
#%% Function for loading a pkl file
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)
#%%
def create_mnist_pair():
    mnist = fetch_mldata('MNIST original')

    X = mnist['data']
    X = np.reshape(X, (70000, 28, 28))
    y = mnist['target']
    N = X.shape[0]

    
    nb_train = 20000
    nb_val = 2000
    nb_test = 5000
    nb_samples = nb_train + nb_val + nb_test
    
    X_sample = np.zeros((nb_samples, 2, 28, 28), dtype = np.uint8)
    y_sample = np.zeros((nb_samples, ), dtype = np.uint8)
    
    count_pos = 0
    count_neq = 0
    already_taken = [ ]
    idx = 0
    print('Generating positive samples')
    while count_pos < nb_samples / 2:
        i, j = np.random.randint(0, N, 2)
        if (i,j) not in already_taken and y[i] == y[j]:
            # Make sure to not take the same again
            already_taken.append((i,j))
            already_taken.append((j,i))
            
            # Save images
            X_sample[idx, 0] = X[i]
            X_sample[idx, 1] = X[j]
            y_sample[idx] = 1
 
            # Update
            count_pos += 1
            idx += 1      
    print('Generation negative samples')
    while count_neq < nb_samples / 2:
        i, j = np.random.randint(0, N, 2)
        if (i,j) not in already_taken and y[i] != y[j]:
            # Make sure to not take the same again
            already_taken.append((i,j))
            already_taken.append((j,i))
            
            # Save images
            X_sample[idx, 0] = X[i]
            X_sample[idx, 1] = X[j]
            y_sample[idx] = 0
 
            # Update
            count_neq += 1
            idx += 1      
    
    # Devide the samples
    print('Creating datasets')
    X_train = np.concatenate((X_sample[:int(nb_train/2)], X_sample[int(nb_samples/2):int((nb_samples+nb_train)/2)]))
    y_train = np.concatenate((y_sample[:int(nb_train/2)], y_sample[int(nb_samples/2):int((nb_samples+nb_train)/2)]))
    X_val =   np.concatenate((X_sample[int(nb_train/2):int((nb_train+nb_val)/2)], X_sample[int((nb_samples+nb_train)/2):int((nb_samples+nb_train+nb_val)/2)]))
    y_val =   np.concatenate((y_sample[int(nb_train/2):int((nb_train+nb_val)/2)], y_sample[int((nb_samples+nb_train)/2):int((nb_samples+nb_train+nb_val)/2)]))
    X_test =  np.concatenate((X_sample[int((nb_train+nb_val)/2):int(nb_samples/2)], X_sample[int((nb_samples+nb_train+nb_val)/2):]))
    y_test =  np.concatenate((y_sample[int((nb_train+nb_val)/2):int(nb_samples/2)], y_sample[int((nb_samples+nb_train+nb_val)/2):]))
    
    # Permute
    print('Permuting datasets')
    perm = np.random.permutation(nb_train)
    X_train = X_train[perm]
    y_train = y_train[perm]

    perm = np.random.permutation(nb_val)
    X_val = X_val[perm]
    y_val = y_val[perm]

    perm = np.random.permutation(nb_test)
    X_test = X_test[perm]
    y_test = y_test[perm]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

#%%
def tower_network(reuse = False):
    net = tfl.input_data(shape = (None, 28, 28, 1))
    
    net = tfl.conv_2d(net, 32, 3, activation = 'relu', reuse = reuse, scope = 'conv1')
    net = tfl.max_pool_2d(net, 2, strides = 2)
    net = tfl.batch_normalization(net)
#    net = tfl.dropout(net, 0.5)
    
    net = tfl.conv_2d(net, 32, 3, activation = 'relu', reuse = reuse, scope = 'conv2')
    net = tfl.max_pool_2d(net, 2, strides = 2)
    net = tfl.batch_normalization(net)
#    net = tfl.dropout(net, 0.5)
    
    net = tfl.fully_connected(net, 512, activation = 'relu', reuse = reuse, scope = 'fc1')
#    net = tfl.dropout(net, 0.5)
    
    return net
#%%    
def similarity_network(tower1, tower2):
    num_classes = 2
    # Marge layer
    net = tfl.merge([tower1, tower2], mode = 'concat', axis = 1, name = 'Merge')
    
    # Decision network
    net = tfl.fully_connected(net, 2048, activation = 'relu')
    net = tfl.dropout(net, 0.5)
    net = tfl.fully_connected(net, 2048, activation = 'relu')
    net = tfl.dropout(net, 0.5)
    
    # Softmax layer
    net = tfl.fully_connected(net, num_classes, activation = 'softmax')
    
    return net
    
    
#%%
if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(description='''This program will train a 
                siamese convolutional neural network on the lfw dataset.''')
    parser.add_argument('-lr', action="store", dest="learning_rate", type=float, default = 0.000001,
                        help = '''Learning rate for optimizer''')
    parser.add_argument('-ne', action="store", dest="num_epochs", type=int, default = 10,
                        help = '''Number of epochs''')
    parser.add_argument('-bs', action="store", dest="batch_size", type=int, default = 100,
                        help = '''Batch size''')
    res = parser.parse_args()
    
    learning_rate = res.learning_rate
    num_epochs = res.num_epochs
    batch_size = res.batch_size
    
    print("Fitting siamese network with parameters")
    print("    with learning rate:     ", learning_rate)
    print("    with batch size:        ", batch_size)  
    print("    in number of epochs:    ", num_epochs)
    
    # Load data
    print('Loading mnist data')
    if not f_exist('mnist_data.pkl'):
        X_train, y_train, X_val, y_val, X_test, y_test = create_mnist_pair()
        save_obj([X_train, y_train, X_val, y_val, X_test, y_test], 'mnist_data')
    X_train, y_train, X_val, y_val, X_test, y_test = load_obj('mnist_data')

    # Tower networks
    net1 = tower_network(reuse = False)
    net2 = tower_network(reuse = True)
    
    # Similarity network
    net = similarity_network(net1, net2)
    
    # Learning algorithm
    net = tfl.regression(net, 
                         optimizer = 'adam', 
                         learning_rate = learning_rate,
                         loss = 'categorical_crossentropy', 
                         name = 'target')
    
    # Training
    model = tfl.DNN(net, tensorboard_verbose = 0,
                    tensorboard_dir='/home/nicki/Documents/CPAB_data_augmentation/mnist_res/')

    uniq_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
    start_time = time.time()
    model.fit(  [X_train[:,0], X_train[:,1]], tfl.data_utils.to_categorical(y_train,2), 
                validation_set = ([X_val[:,0], X_val[:,1]], tfl.data_utils.to_categorical(y_val,2)),
                n_epoch = num_epochs,
                show_metric = True,
                batch_size = batch_size,
                run_id = uniq_id)
    end_time = time.time()
    
    # Do final test evaluation
    score = model.evaluate([X_test[:,0], X_test[:,1]], tfl.data_utils.to_categorical(y_test,2))[0]
    
    print('Mean test acc.: ', score, '. Total time: ', end_time - start_time)

    

