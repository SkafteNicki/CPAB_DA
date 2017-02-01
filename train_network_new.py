"""
Created on Tue Jan 31 13:00:42 2017

@author: Nicki
"""

#%%
import tflearn as tfl
import h5py
import argparse
#%%
def tower_network(reuse = False):
    net = tfl.input_data(shape = (None, 250, 250, 3))
    
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
#    net = tfl.dropout(net, 0.5)
    net = tfl.fully_connected(net, 2048, activation = 'relu')
    
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
    parser.add_argument('-lr', action="store", dest="learning_rate", type=float, default = 0.00001,
                        help = '''Learning rate for optimizer''')
    parser.add_argument('-ne', action="store", dest="num_epochs", type=int, default = 10,
                        help = '''Number of epochs''')
    res = parser.parse_args()
    
    augment_type = res.augment_type
    learning_rate = res.learning_rate
    num_epochs = res.num_epochs
    print("Fitting siamese network with parameters")
    print("    with augmentation type: ", augment_type)
    print("    with learning rate: ", learning_rate)
    print("    in number of epochs: ", num_epochs)
    
    # Load data ....
    if augment_type == 0:
        h5f = h5py.File('lfw_augment_no.h5', 'r')
    elif augment_type == 1:
        h5f = h5py.File('lfw_augment_normal.h5', 'r')
    elif augment_type == 2:
        h5f = h5py.File('lfw_augment_cpab.h5', 'r')
    else:
        ValueError('Set augment type to 0, 1 or 2')
    
    X_train = h5f['X_train']
    y_train = h5f['y_train']
    X_val = h5f['X_val']
    y_val = h5f['y_val']
    X_test = h5f['X_test']
    y_test = h5f['y_test']
    
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
    model = tfl.DNN(net, tensorboard_verbose = 0)
    '''
    tensorboard_verbose:
        0: Loss, Accuracy (Best Speed).
        1: Loss, Accuracy, Gradients.
        2: Loss, Accuracy, Gradients, Weights.
        3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.(Best visualization)
    '''
    model.fit(  [X_train[:,0], X_train[:,1]], tfl.data_utils.to_categorical(y_train,2), 
                validation_set = ([X_val[:,0], X_val[:,1]], tfl.data_utils.to_categorical(y_val,2)),
                n_epoch = num_epochs,
                show_metric = True,
                batch_size = 250,
                run_id = 'lfw_' + str(augment_type))
    
    # Do final test evaluation
    score=10*[0]
    for i in range(10):
        score[i] = model.evaluate([X_test[i,:,0], X_test[i,:,1]], tfl.data_utils.to_categorical(y_test[i],2))[0]
    print('Mean test acc.: ', sum(score) / len(score))
    
    # Close file
    h5f.close()
