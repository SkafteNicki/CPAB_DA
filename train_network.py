"""
Created on Tue Jan 31 13:00:42 2017

@author: Nicki
"""

#%%
import tflearn as tfl
import h5py
import argparse
from datetime import datetime
import time
#%%
def tower_network(reuse = False, wd = 0.001):
    net = tfl.input_data(shape = (None, 125, 125, 3))
    
    net = tfl.conv_2d(net, 32, 11, activation = 'relu', reuse = reuse, 
                      scope = 'conv1', weight_decay = wd)
    net = tfl.max_pool_2d(net, 3)
    net = tfl.batch_normalization(net)
    
    net = tfl.conv_2d(net, 16, 9, activation = 'relu', reuse = reuse, 
                      scope = 'conv2', weight_decay = wd)
    net = tfl.max_pool_2d(net, 2)
    net = tfl.batch_normalization(net)
    
    net = tfl.fully_connected(net, 1024, activation = 'relu', reuse = reuse, 
                              scope = 'fc1', weight_decay = wd)
    
    return net
#%%    
def similarity_network(tower1, tower2, wd = 0.001):
    num_classes = 2
    # Marge layer
    net = tfl.merge([tower1, tower2], mode = 'concat', axis = 1, name = 'Merge')
    
    # Decision network
    net = tfl.fully_connected(net, 1024, activation = 'relu', regularizer = 'L2',
                              scope = 'fc2', weight_decay = wd)
    net = tfl.fully_connected(net, 1024, activation = 'relu', regularizer = 'L2',
                              scope = 'fc3', weight_decay = wd)
#    
    # Softmax layer
    net = tfl.fully_connected(net, num_classes, activation = 'softmax')
    
    return net
#%%
if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser(description='''This program will train a 
                siamese convolutional neural network on the lfw dataset.''')
    parser.add_argument('-at', action="store", dest="augment_type", type = int, default = 0,
                        help = '''Augmentation type. 0=no augmentation, 1=normal augmentation
                                ,2=cpab augmentation''')
    parser.add_argument('-lr', action="store", dest="learning_rate", type=float, default = 0.000001,
                        help = '''Learning rate for optimizer''')
    parser.add_argument('-ne', action="store", dest="num_epochs", type=int, default = 10,
                        help = '''Number of epochs''')
    parser.add_argument('-bs', action="store", dest="batch_size", type=int, default = 100,
                        help = '''Batch size''')
    parser.add_argument('-wd', action="store", dest="weight_decay", type=float, default = 0.0001,
                        help = '''Weight decay''')
    res = parser.parse_args()
    
    augment_type = res.augment_type
    learning_rate = res.learning_rate
    num_epochs = res.num_epochs
    batch_size = res.batch_size
    weight_decay = res.weight_decay
    
    print("Fitting siamese network with parameters")
    print("    with augmentation type: " + str(augment_type))
    print("    with learning rate:     " + str(learning_rate))
    print("    with batch size:        " + str(batch_size))
    print("    with weight decay:      " + str(weight_decay))
    print("    in number of epochs:    " + str(num_epochs))
    
    # Load data ....
    if augment_type == 0:
        h5f = h5py.File('datasets/lfw_augment_no_cv_0.h5', 'r')
#    elif augment_type == 1:
#        h5f = h5py.File('lfw_augment_normal.h5', 'r')
#    elif augment_type == 2:
#        h5f = h5py.File('lfw_augment_cpab.h5', 'r')
#    else:
#        ValueError('Set augment type to 0, 1 or 2')
#    
    X_train = h5f['X_train']
    y_train = h5f['y_train']
    X_test = h5f['X_test']
    y_test = h5f['y_test']
    
    # Tower networks
    net1 = tower_network(reuse = False, wd = weight_decay)
    net2 = tower_network(reuse = True, wd = weight_decay)
    
    # Similarity network
    net = similarity_network(net1, net2, wd = weight_decay)
    
    # Learning algorithm
    net = tfl.regression(net, 
                         optimizer = 'adam', 
                         learning_rate = learning_rate,
                         loss = 'categorical_crossentropy', 
                         name = 'target',
                         to_one_hot = True,
                         n_classes = 2)
    
    # Training
    model = tfl.DNN(net, tensorboard_verbose = 3,
                    tensorboard_dir='/home/nicki/Documents/CPAB_data_augmentation/network_res/')
    
    uniq_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
    start_time = time.time()
    model.fit(  [X_train[:,0], X_train[:,1]], y_train, 
                validation_set = ([X_test[:,0], X_test[:,1]], y_test),
                n_epoch = num_epochs,
                show_metric = True,
                batch_size = batch_size,
                run_id = 'lfw_' + str(augment_type) + '_' + uniq_id)
    end_time = time.time()
        
    # Close file
    h5f.close()
