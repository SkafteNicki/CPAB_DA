"""
Created on Thu Dec  1 21:15:21 2016
@author: nicki
"""
#%% Pakages to import
from utils import load_lfw_pairs, load_obj, save_obj
import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.layers import DropoutLayer, batch_norm, ConcatLayer
from lasagne.nonlinearities import rectify, softmax
from collections import OrderedDict
import cv2
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
    
    
    
#%%
def create_network(input_shape, n_layers, n_filters, n_hid):
    net = OrderedDict()
    
    # Input layers
    net['input_0'] = InputLayer(shape=input_shape)
    net['input_1'] = InputLayer(shape=input_shape)
    
    # Conv layers + maxpool + dropout 1    
    net['conv_0_0'] = Conv2DLayer(      net['input_0'], num_filters = n_filters, filter_size = 3, pad = 1, nonlinearity = rectify)
    net['conv_1_0'] = Conv2DLayer(      net['input_1'], num_filters = n_filters, filter_size = 3, pad = 1, nonlinearity = rectify, 
                                    W = net['conv_0_0'].W, b = net['conv_0_0'].b)
    net['pool_0_0'] = MaxPool2DLayer(   net['conv_0_0'], pool_size = [2,2])
    net['pool_1_0'] = MaxPool2DLayer(   net['conv_1_0'], pool_size = [2,2])
    net['norm_0_0'] = batch_norm(       net['pool_0_0'])
    net['norm_1_0'] = batch_norm(       net['pool_1_0'])    
    net['drop_0_0'] = DropoutLayer(     net['norm_0_0'], p = 0.5)
    net['drop_1_0'] = DropoutLayer(     net['norm_1_0'], p = 0.5)    
    
    for n in range(1, n_layers):
        # Conv layers + maxpool + dropout 2    
        net['conv_0_' + str(n)] = Conv2DLayer(      net['drop_0_' + str(n-1)], num_filters = n_filters, filter_size = 3, pad = 1, nonlinearity = rectify)
        net['conv_1_' + str(n)] = Conv2DLayer(      net['drop_1_' + str(n-1)], num_filters = n_filters, filter_size = 3, pad = 1, nonlinearity = rectify, 
                                                W = net['conv_0_' + str(n)].W, b = net['conv_0_' + str(n)].b) # this is the siemese connection                                             
        net['pool_0_' + str(n)] = MaxPool2DLayer(   net['conv_0_' + str(n)], pool_size = [2,2])
        net['pool_1_' + str(n)] = MaxPool2DLayer(   net['conv_1_' + str(n)], pool_size = [2,2])
        net['norm_0_' + str(n)] = batch_norm(       net['pool_0_' + str(n)])
        net['norm_1_' + str(n)] = batch_norm(       net['pool_1_' + str(n)])
        net['drop_0_' + str(n)] = DropoutLayer(     net['norm_0_' + str(n)], p = 0.5)
        net['drop_1_' + str(n)] = DropoutLayer(     net['norm_1_' + str(n)], p = 0.5)    
    
    # Merge the siemese network
    net['merge'] = ConcatLayer([net['drop_0_' + str(n_layers-1)], net['drop_1_' + str(n_layers-1)]])
    
    # End with decision network
    net['hid_0'] = DenseLayer(net['merge'], num_units = n_hid, nonlinearity = rectify)
    net['hid_1'] = DenseLayer(net['hid_0'], num_units = n_hid, nonlinearity = rectify)
    net['output'] = DenseLayer(net['hid_1'], num_units = 2, nonlinearity = softmax)    
    
    return net
#%%
def compile_network(net):
    # Setting up the graph in theano
    sym_x1 = T.tensor4()
    sym_x2 = T.tensor4()
    sym_t = T.ivector()
    learning_rate = T.scalar(name='learning_rate')

    # Retrive network output      
    input_layers = {net['input_0']: sym_x1, net['input_1']: sym_x2}    
    train_out = lasagne.layers.get_output(net['output'], input_layers, deterministic = False)
    eval_out = lasagne.layers.get_output(net['output'], input_layers, deterministic = True)

    # Get list of all trainable parameters in network
    all_params = lasagne.layers.get_all_params(net['output'], trainable=True)

    # Get the cost
    cost = T.nnet.categorical_crossentropy(train_out + 1e-8, sym_t).mean()

    # Get all gradients for training
    all_grads = T.grad(cost, all_params)

    # Set update function
    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=learning_rate)

    # Define theano functions for training and evaluation of network    
    f_eval = theano.function([sym_x1, sym_x2], eval_out, allow_input_downcast=True)
    f_train = theano.function([sym_x1, sym_x2, sym_t, learning_rate], cost, updates = updates, allow_input_downcast=True)
    
    return f_train, f_eval
#%%
def get_batch_data(n_scaling, X, y, trans, Nk):
    if n_scaling == 1:
        return np.rollaxis(X, 4, 2), y
    else:
        K = len(trans)
        n_trans = len(trans[0])
        new_X = np.zeros(shape=(n_scaling*len(X), X.shape[1], X.shape[4], X.shape[2], X.shape[3]), dtype=np.uint8)
        new_y = np.zeros(shape=(n_scaling*len(y)), dtype=np.uint8)
        count = 0
        for i in range(X.shape[0]):
            new_X[count] = np.rollaxis(X[i], 3, 1)
            new_y[count] = y[i]
            count += 1
            
            image = X[i]
            for n in range(n_scaling-1):
                # Draw random cluster index
                cluster = np.random.choice(range(K), size = 1, p = Nk / Nk.sum())[0]

                for j in range(2):
                    # Draw a random generated transformer
                    rand_trans = np.random.randint(0, n_trans)                
                
                    # Get random transformation from cluster
                    pts_fwd = trans[cluster][rand_trans]
                
                    # Do image deformation                
                    map1 = pts_fwd[:,0].astype(np.float32).reshape(image[j].shape[:2])
                    map2 = pts_fwd[:,1].astype(np.float32).reshape(image[j].shape[:2])
                    new_img = np.zeros_like(image[j])
                    cv2.remap(src = image[j], map1 = map1, map2 = map2, interpolation = cv2.INTER_LANCZOS4, dst = new_img)
                    new_X[count,j] = np.rollaxis(new_img,2)
                    
                new_y[count] = y[i]
                count += 1
				                       
        return new_X, new_y
    
#%% Main scrip
if __name__ == '__main__':
    # Load lfw data
    imgs, index, pairs = load_lfw_pairs()
    
    # Load estimated cluster parameters    
    _, _, _, Nk, _ = load_obj('cluster_data/cluster_parameters_processed') 

    # Create data
    print("Creating data")
    X_train, y_train, X_val, y_val, X_test, y_test = create_img_pairs(imgs, index, pairs)
    
    # Parameters
    n_scaling = 1   
    n_layer = 4
    n_filt = 40
    n_hid = 1000
    n_epochs = 300
    lr_base = 0.1
    lr_decay = 0.99    
    
    # Load presampled transformations
    trans = load_obj('transformations')
    
    # Create network
    print("Setting up network")
    nchannels, rows, cols = 3, 250, 250
    net = create_network((None, nchannels, rows, cols), n_layer, n_filt, n_hid)
    
    # Compile network
    print("Compiling network")
    f_train, f_eval = compile_network(net)
    
    # Training parameters
    max_batch_size = 10
    batch_size = int(np.floor(max_batch_size * 1.0/n_scaling))
    train_size = X_train.shape[0]
    val_size = X_val.shape[0]
    test_size = X_test.shape[1]
    num_batch_train = train_size // batch_size
    num_batch_val = val_size // batch_size
    num_batch_test = test_size // batch_size
    
    print("Training network")
    loss, train_acc, val_acc = [ ], [ ], [ ]
    try:
        for e in range(n_epochs):
            # Backpropagate training
            cur_loss = 0
            lr = lr_base #* (lr_decay ** e)
            for b in range(num_batch_train):
                batch_idx = range(b*batch_size, (b+1)*batch_size)
                batch_data, batch_target = get_batch_data(n_scaling,
                    X_train[batch_idx], y_train[batch_idx], trans, Nk)
                cur_loss += f_train(batch_data[:,0], batch_data[:,1], batch_target, lr)
            loss += [cur_loss / num_batch_train]

            # Feedforward training
            cur_train_acc = 0
            for b in range(num_batch_train):
                batch_idx = range(b*batch_size, (b+1)*batch_size)
                batch_data, batch_target = get_batch_data(1,
                    X_train[batch_idx], y_train[batch_idx], trans, Nk)
                net_out = f_eval(batch_data[:,0], batch_data[:,1])
                preds = np.argmax(net_out, axis=1)
                cur_train_acc += np.sum(preds == batch_target) * 1.0/preds.size
            train_acc += [cur_train_acc / num_batch_train]
        	
            # Feedforward validation
            cur_val_acc = 0
            for b in range(num_batch_val):
                batch_idx = range(b*batch_size, (b+1)*batch_size)
                batch_data, batch_target = get_batch_data(1,
                    X_val[batch_idx], y_val[batch_idx], trans, Nk)
                net_out = f_eval(batch_data[:,0], batch_data[:,1])
                preds = np.argmax(net_out, axis=1)
                cur_val_acc += np.sum(preds == batch_target) * 1.0/preds.size
            val_acc += [cur_val_acc / num_batch_val]
			
            # Print progress
            print("Epoch: {}, Loss: {}, Train acc.: {}, Val acc.: {}".format(
			e+1, round(loss[-1], 5), round(train_acc[-1], 3), round(val_acc[-1], 3)))
        
    except KeyboardInterrupt:
        print("Training interupted")
    finally:
        print("Computing final test error")
#        # Do this for all the 10 test folds
#        test_acc = np.zeros(shape=(10))
#        for i in range(10):
#            cur_test_acc = 0
#            for b in range(num_batch_test):
#                batch_idx = range(b*batch_size, (b+1)*batch_size)
#                batch_data, batch_target = get_batch_data(1,
#                    X_test[batch_idx], y_test[batch_idx], trans, Nk)
#                preds = (net_out[:,0] > 0.5).astype('int')
#                cur_test_acc += np.sum(net_out == preds)
#            test_acc[i] = cur_test_acc / (num_batch_test * batch_size)
#        save_obj([e+1, loss, train_acc, val_acc, test_acc], 'network_results/pair' + str(n_scaling))
    
    
            
    
