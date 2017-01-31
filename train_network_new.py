"""
Created on Tue Jan 31 13:00:42 2017

@author: Nicki
"""

#%%
import tflearn as tfl

#%%
def tower_network(reuse = False, augment = False):
    if augment:
        data_aug = tfl.ImageAugmentation()
        data_aug.add_random_flip_leftright()
        data_aug.add_random_rotation(40)
        data_aug.add_random_blur()
        net = tfl.input_data(shape = (None), data_augmentation=data_aug)    
    else:
        net = tfl.input_data(shape = (None, 250, 250, 3))
    
    net = tfl.conv_2d(net, 32, 3, activation = 'relu', reuse = reuse, scope = 'conv1')
    net = tfl.max_pool_2d(net, 2, strides = 2)
    net = tfl.dropout(net, 0.5)
    
    net = tfl.conv_2d(net, 32, 3, activation = 'relu', reuse = reuse, scope = 'conv2')
    net = tfl.max_pool_2d(net, 2, strides = 2)
    net = tfl.dropout(net, 0.5)
    
    net = tfl.fully_connected(net, 512, activation = 'relu', reuse = reuse, scope = 'fc1')
    net = tfl.dropout(net, 0.5)
    
    return net
#%%    
def similarity_network(tower1, tower2):
    num_classes = 2
    # Marge layer
    net = tfl.merge([tower1, tower2], mode = 'concat', axis = 1, name = 'Merge')
    
    # Decision network
    net = tfl.fully_connected(net, 2048, activation = 'relu')
    net = tfl.dropout(net, 0.5)
    net = tfl.fully_connected(net, 2028, activation = 'relu')
    
    # Softmax layer
    net = tfl.fully_connected(net, num_classes, activation = 'softmax')
    
    return net
#%%
if __name__ == '__main__':
    # Parameters
    augment_type = 0
    
    
    # Load data ....
    # ...
    # ...
    # ....
    
    # Tower networks
    net1 = tower_network(reuse = False, augment = augment_type)
    net2 = tower_network(reuse = True, augment = augment_type)
    
    # Similarity network
    net = similarity_network(net1, net2)
    
    # Learning algorithm
    net = tfl.regression(net, optimizer = 'adam', learning_rate = 0.01,
                         loss = 'categorical_crossentropy', name = 'target')
    
    # Training
    model = tfl.DNN(net, tensorboard_verbose = 0)
    '''
    tensorboard_verbose:
        0: Loss, Accuracy (Best Speed).
        1: Loss, Accuracy, Gradients.
        2: Loss, Accuracy, Gradients, Weights.
        3: Loss, Accuracy, Gradients, Weights, Activations, Sparsity.(Best visualization)
    '''
    model.fit([X_train[:,0], X_train[:,1]], Y_train, n_epoch = 5,
              validatation_set = ([X_val[:,0], X_val[:,1]], Y_val))
    
    
