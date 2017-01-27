"""
Created on Tue Nov 22 18:36:29 2016
@author: Nicki

Description:
    This script train a neural network on either the lfw data or a augmented 
    version (created at run time) based on the clustering results from the
    fit_clusters.py script
"""
#%% Packages to import
from utils import load_obj, save_obj, f_exist, set_params, load_lfw, get_lfw_network_data
from utils import get_batch_data2, folder_create
import argparse
import numpy as np
from collections import OrderedDict
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer
from lasagne.layers import DropoutLayer, batch_norm, get_all_param_values
from lasagne.nonlinearities import rectify, sigmoid
import time


#%% Function for creating the network
def CreateNetwork(nchannels, rows, cols, n_classes, nlayers=1, n_filters=10, filt_size=5, n_hid = 500):
    # Structure for network
    net = OrderedDict()
    
    # First layer of network
    net['input'] =  InputLayer( shape=(None, nchannels, rows, cols))
    net['conv_0'] = Conv2DLayer(    net['input'], 
                                    num_filters = n_filters, 
                                    filter_size = filt_size, 
                                    pad = 1, 
                                    nonlinearity = rectify)
    net['pool_0'] = MaxPool2DLayer( net['conv_0'], 
                                    pool_size = [2,2])
    net['norm_0'] = batch_norm(     net['pool_0'])
    net['drop_0'] = DropoutLayer(   net['norm_0'], 
                                    p = 0.5, 
                                    rescale = True)
    # Remaining layer of networks
    for n in range(1,nlayers):
        net['conv_' + str(n)] = Conv2DLayer(    net['drop_' + str(n-1)], 
                                                num_filters = n_filters, 
                                                filter_size = filt_size, 
                                                pad = 1, 
                                                nonlinearity = rectify)
        net['pool_' + str(n)] = MaxPool2DLayer( net['conv_' + str(n)], 
                                                pool_size = [2,2])
        net['norm_' + str(n)] = batch_norm(     net['pool_' + str(n)])
        net['drop_' + str(n)] = DropoutLayer(   net['norm_' + str(n)],
                                                p = 0.5,
                                                rescale = True)
    net['dense_0'] =  DenseLayer(   net['drop_' + str(nlayers-1)],
                                    num_units = n_hid,
                                    nonlinearity = rectify)
    net['drop_hid_0'] =DropoutLayer(net['dense_0'],
                                    p = 0.5,
                                    rescale = True)
    net['dense_1'] =  DenseLayer(   net['drop_hid_0'],
                                    num_units = n_hid,
                                    nonlinearity = rectify)
    net['drop_hid_1'] =DropoutLayer(net['dense_1'],
                                    p = 0.5,
                                    rescale = True)
    net['output'] = DenseLayer(     net['drop_hid_1'], 
                                    num_units = n_classes, 
                                    nonlinearity = sigmoid)
    return net

#%% Function for compiling the network
def compile_network(net):
    sym_x = T.tensor4()
    sym_t = T.imatrix()

    # Retrive network output
    train_out = lasagne.layers.get_output(net['output'], sym_x, deterministic = False)
    eval_out = lasagne.layers.get_output(net['output'], sym_x, deterministic = True)

    # Get list of all trainable parameters in network
    all_params = lasagne.layers.get_all_params(net['output'], trainable=True)
       
    # Get the cost
    cost = T.nnet.binary_crossentropy(train_out+1e-8, sym_t).mean() #+ l2_weight * l2_cost
    
    # Get all gradients for training
    all_grads = T.grad(cost, all_params)
    
    # Set update function
    updates = lasagne.updates.adam(all_grads, all_params, learning_rate=0.001)

    # Define theano functions for training and evaluation of network
    f_eval = theano.function([sym_x], eval_out, allow_input_downcast=True)
    f_train = theano.function([sym_x, sym_t], cost, updates = updates, allow_input_downcast=True)
    return f_train, f_eval
    

#%% Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''This program will fit a CNN to
            the lfw data or a augmented version, that tries to predict a number
            of person attributes''')
    parser.add_argument('-n', action="store", dest="n_sample", type = int, default = 1,
                        help = '''Augmentation scale factor. Set to 1, for not augmentation''')
    parser.add_argument('-nl', action="store", dest="n_layer", type = int, default = 1,
                        help = '''Number of conv, pool, norm and drop layers in network''')
    parser.add_argument('-nf', action="store", dest="n_filt", type = int, default = 10,
                        help = '''Number of filters in each conv layer''')
    parser.add_argument('-nh', action="store", dest="n_hid", type = int, default = 500,
                        help = '''Number of neurons in fully connected layers''')
    parser.add_argument('-e', action="store", dest="n_epochs", type = int, default = 200,
                        help = '''Number of epochs to train network''')
    res = parser.parse_args()
    
    n_samples = res.n_sample
    n_layer = res.n_layer
    n_filt = res.n_filt
    n_hid = res.n_hid
    num_epochs = res.n_epochs
    print "Training CNN on lfw"
    print "  with augment scaling:     ", n_samples
    print "  with number of layers:    ", n_layer
    print "  with number of conv filt: ", n_filt
    print "  with number of hid neurons", n_hid
    print "  in number of epochs:      ", num_epochs
                        
    # Parameters from theta estimation
    if not f_exist('params.pkl'):
        set_params()
    params = load_obj('params')
      
    # Load the LFW dataset
    imgs, index, landmarks, attribute_names, attributes, split = load_lfw(str(params['num_landmarks']))
    
    # Load estimated deformations
    thetadata, info, theta_persons = load_obj('cluster_data/theta_data')
    
    # Load estimated cluster parameters    
    _, _, _, Nk, r_nk = load_obj('cluster_data/cluster_parameters_processed')
    
    # Load presampled transformations
    print "Loading pretrained transformations"
    trans = load_obj('transformations')
    
    # Create a modified version of lfw, that better suits the training of the network
    data_folder = 'datasets'
    if not f_exist(data_folder + '/' + 'network_train_data.pkl'):
        print "Creating training set"
        lfw_img_tr, lfw_lm_tr, lfw_att_tr, lfw_pri_tr = get_lfw_network_data('train', imgs, 
            landmarks, index, attribute_names, attributes, split, theta_persons, Nk, r_nk)
        save_obj([lfw_img_tr, lfw_lm_tr, lfw_att_tr, lfw_pri_tr], 
                 data_folder + '/' + 'network_train_data')
    else:
        print "Loading training set"
        lfw_img_tr, lfw_lm_tr, lfw_att_tr, lfw_pri_tr = load_obj(data_folder + '/' + 'network_train_data')
    
    if not f_exist(data_folder + '/' + 'network_val_data.pkl'):
        print "Creating validation set"
        lfw_img_val, lfw_lm_val, lfw_att_val, lfw_pri_val = get_lfw_network_data('val', imgs,   
            landmarks, index, attribute_names, attributes, split, theta_persons, Nk, r_nk)
        save_obj([lfw_img_val, lfw_lm_val, lfw_att_val, lfw_pri_val], 
                 data_folder + '/' + 'network_val_data')
    else:
        print "Loading validation set"
        lfw_img_val, lfw_lm_val, lfw_att_val, lfw_pri_val = load_obj(data_folder + '/' + 'network_val_data')
    
    if not f_exist(data_folder + '/' + 'network_test_data.pkl'):
        print "Creating test set"
        lfw_img_te, lfw_lm_te, lfw_att_te, lfw_pri_te = get_lfw_network_data('test', imgs,   
            landmarks, index, attribute_names, attributes, split, theta_persons, Nk, r_nk)
        save_obj([lfw_img_te, lfw_lm_te, lfw_att_te, lfw_pri_te], 
                 data_folder + '/' + 'network_test_data')
    else:
        print "Loading test set"
        lfw_img_te, lfw_lm_te, lfw_att_te, lfw_pri_te = load_obj(data_folder + '/' + 'network_test_data')
    
    # Free space
    del imgs, landmarks, index, attribute_names, attributes, theta_persons, info
    
    # Create network
    print "Creating network"
    shape = (lfw_img_tr.shape[3], lfw_img_tr.shape[1], lfw_img_tr.shape[2]) 
    n_attributes = lfw_att_tr.shape[1]
    net = CreateNetwork(shape[0], shape[1], shape[2], n_attributes, 
                        nlayers=n_layer, n_filters=n_filt, filt_size=3, n_hid = n_hid)    
    # Compile network    
    print "Compiling network"
    f_train, f_eval = compile_network(net)
    
    # Traning part
    max_batch_size = 10 # if the batch size becomes larger than this, then the
                        # complexity of the network needs to be reduced
    batch_size = int(np.floor(max_batch_size * 1.0/n_samples))
    
    num_train = lfw_img_tr.shape[0]
    num_val = lfw_img_val.shape[0]
    num_batch_train = num_train // batch_size
    num_batch_val = num_val // batch_size
    
    train_acc, val_acc, loss = [ ], [ ], [ ]
    print "Running for: n_samples = " + str(n_samples) + " and epoch = " + str(num_epochs)
    try:
        tstart = time.time()
        for e in range(num_epochs):
            cur_loss = 0
            # Train parameters
            for b in range(num_batch_train):
                batch_idx = range(b*batch_size, (b+1)*batch_size)
                batch_imgs, batch_targets = get_batch_data2(n_samples, lfw_img_tr[batch_idx], 
                    lfw_att_tr[batch_idx], lfw_pri_tr[batch_idx], trans)
                batch_loss = f_train(batch_imgs, batch_targets)
                cur_loss += batch_loss
            loss += [cur_loss / num_batch_train]
            
            train_acc_cur = 0
            for b in range(num_batch_train):
                batch_idx = range(b*batch_size, (b+1)*batch_size)
                batch_imgs, batch_targets = get_batch_data2(n_samples, lfw_img_tr[batch_idx], 
                    lfw_att_tr[batch_idx], lfw_pri_tr[batch_idx], trans)
                net_out = f_eval(batch_imgs)
                preds = (net_out>0.5).astype('uint8')
                train_acc_cur += 1.0/net_out.size * np.sum(preds == batch_targets)
            train_acc += [train_acc_cur / num_batch_train]    
            
            val_acc_cur = 0                        
            for b in range(num_batch_val):
                batch_idx = range(b*batch_size, (b+1)*batch_size)
                batch_imgs, batch_targets = get_batch_data2(1, lfw_img_tr[batch_idx], 
                    lfw_att_tr[batch_idx], lfw_pri_tr[batch_idx], trans)
                net_out = f_eval(batch_imgs)
                preds = (net_out>0.5).astype('uint8')
                val_acc_cur += 1.0/net_out.size * np.sum(preds == batch_targets)
            val_acc += [val_acc_cur / num_batch_val]
            
            # Print how the network is preforming            
            print "Epoch {}: Train loss {}, Train acc {}, Val acc {}".format(e+1,
                round(loss[-1],2), round(train_acc[-1],2), round(val_acc[-1],2))
                            
    except KeyboardInterrupt: # Handle keyboard interruption
        print "Network training interrupted"
        
    finally:    # End computations with computing the test accurracy, even if we
                # where interrupted in the training
    
        # Calculate training time
        tstop = time.time()
        tdiff = tstop - tstart
        
        # Calculate test accuracy for each of the 10 test sets   
        test_acc = 10 * [[]]
        for t in range(10):
            num_test = lfw_img_te[t].shape[0]
            num_batch_test = num_test // batch_size
            test_acc_cur = 0
            for b in range(num_batch_test):
                batch_idx = range(b*batch_size, (b+1)*batch_size)
                batch_imgs, batch_targets = get_batch_data2(1, lfw_img_tr[batch_idx], 
                    lfw_att_tr[batch_idx], lfw_pri_tr[batch_idx], trans)
                net_out = f_eval(batch_imgs)
                preds = (net_out>0.5).astype('uint8')
                test_acc_cur += 1.0/net_out.size * np.sum(preds == batch_targets)
            test_acc[t] = test_acc_cur / num_batch_test
        
        # Save all informations to a file
        network_params = {'epochs': num_epochs, 'n_samples': n_samples, 'n_layers': n_layer,
                          'n_filters': n_filt, 'n_hid': n_hid, }
        folder_create('network_results')
        save_obj([e, loss, train_acc, val_acc, test_acc,  network_params, tdiff, 
                  get_all_param_values(net['output'])], 
                  'network_results/res_' + str(n_samples) + '_' + str(n_layer))

