"""
Created on Tue Jan 31 13:00:42 2017

@author: Nicki
"""

#%%
import tflearn as tfl
import tensorflow as tf
import h5py
import argparse
from datetime import datetime
import time
#%%
def tower_network(reuse = False, dr = 0.5, wd = 0.001, nc = 2, nf = [8, 16], fs = [11, 9]):
    net = tfl.input_data(shape = (None, 125, 125, 3))
    
    for n in range(nc):
        net = tfl.conv_2d(net, nf[n], fs[n], activation = 'relu', reuse = reuse,
                          scope = 'conv' + str(n))#, regularizer = 'L2',
                          #weight_decay = wd)
        if n < 2:
            net = tfl.max_pool_2d(net, 3)
            
        net = tfl.batch_normalization(net)
        net = tfl.dropout(net, dr)
    
    return net
#%%    
def similarity_network(tower1, tower2, wd = 0.001, nfc = 2, nn = [1024, 1024]):
    num_classes = 2
    # Marge layer
    net = tf.abs(tower1 - tower2)
    
    for n in range(nfc):
        net = tfl.fully_connected(net, nn[n], activation = 'relu', regularizer = 'L2',
                                  scope = 'fc' + str(n), weight_decay = wd)  
    
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
                        help = '''Weight decay, set to 0 for no L2 regulization''')
    parser.add_argument('-dr', action="store", dest="dropout_rate", type=float, default = 0.5,
                        help = '''Dropout rate, set to 1 for no dropout''')
    parser.add_argument('-fs', action="store", dest="filt_size", type=int, nargs = '+',
                        default = [11, 9], help = '''Filter size in conv layers''')
    parser.add_argument('-nf', action="store", dest="num_filt", type=int, nargs = '+',
                        default = [8, 16], help = '''Number of filters in conv layers''')
    parser.add_argument('-nc', action="store", dest="num_conv_layers", type=int, default = 2,
                        help = '''Number of conv layers''')
    parser.add_argument('-nfc', action="store", dest="num_fully_layers", type=int, default = 2,
                        help = '''Number of fully connected layers''')
    parser.add_argument('-nn', action="store", dest="num_neurons", type=int, nargs = '+',
                        default = [1024, 1024], help = '''Number of neurons in each fully_connected layer''')
    res = parser.parse_args()
    
    augment_type = res.augment_type
    learning_rate = res.learning_rate
    num_epochs = res.num_epochs
    batch_size = res.batch_size
    weight_decay = res.weight_decay
    dropout_rate = res.dropout_rate
    filt_size = res.filt_size
    num_filt = res.num_filt
    num_conv_layers = res.num_conv_layers
    num_fully_layers = res.num_fully_layers
    num_neurons = res.num_neurons
    assert len(filt_size) == num_conv_layers
    assert len(num_filt) == num_conv_layers
    assert len(num_neurons) == num_fully_layers
              
              
    print("Fitting siamese network with parameters")
    print("    with augmentation type:       " + str(augment_type))
    print("    with number of conv layers:   " + str(num_conv_layers))
    print("    with number of filters:       " + str(num_filt))
    print("    with filter sizes:            " + str(filt_size))
    print("    with number of fully layers:  " + str(num_fully_layers))
    print("    with number of neurons:       " + str(num_neurons))
    print("    with learning rate:           " + str(learning_rate))
    print("    with batch size:              " + str(batch_size))
    print("    with weight decay:            " + str(weight_decay))
    print("    with dropout rate:            " + str(dropout_rate))
    print("    in number of epochs:          " + str(num_epochs))
    
    # Load data ....
    if augment_type == 0:
        h5f = h5py.File('datasets/lfw_0_down/lfw_augment_no_cv_0.h5', 'r')
    elif augment_type == 1:
        h5f = h5py.File('datasets/lfw_1_down/lfw_augment_normal_cv_0.h5', 'r')
    elif augment_type == 2:
        h5f = h5py.File('datasets/lfw_2_down/lfw_augment_cpab_cv_0.h5', 'r')
    else:
        ValueError('Set augment type to 0, 1 or 2')
    
    X_train = h5f['X_train']
    y_train = h5f['y_train']
    X_test = h5f['X_test']
    y_test = h5f['y_test']
    
    # Tower networks
    net1 = tower_network(reuse = False, wd = weight_decay, dr = dropout_rate,
                         nc = num_conv_layers, nf = num_filt, fs = filt_size)
    net2 = tower_network(reuse = True, wd = weight_decay, dr = dropout_rate, 
                         nc = num_conv_layers, nf = num_filt, fs = filt_size)
    
    # Similarity network
    net = similarity_network(net1, net2, wd = weight_decay, 
                             nfc = num_fully_layers, nn = num_neurons)
    
    # Learning algorithm
    net = tfl.regression(net, 
                         optimizer = 'adam', 
                         learning_rate = learning_rate,
                         loss = 'categorical_crossentropy', 
                         name = 'target',
                         to_one_hot = True,
                         n_classes = 2)
    
    # Training
    model = tfl.DNN(net, tensorboard_verbose = 0,
                    tensorboard_dir = '/home/nicki/Documents/CPAB_data_augmentation/network_res/',
                    best_checkpoint_path = '/home/nicki/Documents/CPAB_data_augmentation/best_model/',
                    best_val_accuracy = 0.8)
    
    uniq_id = datetime.now().strftime('%Y_%m_%d_%H_%M')
    start_time = time.time()
    model.fit(  [X_train[:,0], X_train[:,1]], y_train, 
                validation_set = ([X_test[:,0], X_test[:,1]], y_test),
                n_epoch = num_epochs,
                show_metric = True,
                batch_size = batch_size,
                run_id = 'lfw_' + str(augment_type) + '_' + uniq_id)
    end_time = time.time()
    #score = model.evaluate([X_test[:,0], X_test[:,1]], y_test)[0]
    #print('Final test score:', score)
    
    # Close file
    h5f.close()
