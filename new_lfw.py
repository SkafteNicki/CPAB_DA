##!/usr/bin/env python2
## -*- coding: utf-8 -*-
#"""
#Created on Wed Feb  8 21:12:01 2017
#
#@author: nicki
#"""
#%%
from utils_func import load_lfw, load_obj, set_params, f_exist, generate_new_images
#from scipy.ndimage import zoom
import numpy as np
import h5py
import imgaug.imgaug.augmenters as iaa

from of.gpu import CpuGpuArray
from cpab.cpa2d.inference.transformation.TransformationFitter import TransformationFitter
from of.utils import Bunch

#%%
if __name__ == '__main__':
    imgs, index, _, pairs = load_lfw()
    rows, cols, nchannels = 250, 250, 3
    
    # Convert to half the resolution
#    imgs_temp = np.zeros((13233, rows, cols, nchannels), dtype = np.uint8)
#    for idx in range(imgs.shape[0]):
#        imgs_temp[idx] = zoom(imgs[idx], (0.5, 0.5, 1), order = 3)
#    imgs = imgs_temp
#    del imgs_temp
#    
    # Extract the 10 folds
    X = np.zeros(shape=(10, len(pairs['test'][0]), 2, rows, cols, nchannels), dtype=np.uint8)
    y = np.zeros(shape=(10, len(pairs['test'][0])), dtype=np.uint8)
    for i in range(10):
        count = 0
        for n in np.random.permutation(len(pairs['test'][i])):
            X[i,count,0] = imgs[index[pairs['test'][i][n][0]][pairs['test'][i][n][1]-1]]
            X[i,count,1] = imgs[index[pairs['test'][i][n][2]][pairs['test'][i][n][3]-1]]   
            y[i,count] = 1 if pairs['test'][i][n][0] == pairs['test'][i][n][2] else 0
            count += 1
    #%%
    # Write to 10 files with correct data
    for i in range(10):
        X_train = np.concatenate(tuple([X[k] for k in range(10) if k != i]))
        y_train = np.concatenate(tuple([y[k] for k in range(10) if k != i]))
        X_test = X[i]
        y_test = y[i]
        
        h5f = h5py.File('datasets/lfw_0/lfw_augment_no_cv_' + str(i) + '.h5', 'w')
        h5f.create_dataset('X_train', data = X_train)
        h5f.create_dataset('y_train', data = y_train)
        h5f.create_dataset('X_test', data = X_test)
        h5f.create_dataset('y_test', data = y_test)
        h5f.close()

    #%%
    n_scaling = 5
    augmenters = [ iaa.Fliplr(0.5),
                   iaa.Affine(rotate = (-45, 45)),
                   iaa.GaussianBlur(sigma=(0, 5.0)),
                   iaa.Crop(percent=(0, 0.1)),
                   iaa.Affine(scale = {"x": (0.8, 1.2), "y": (0.8, 1.2)})]

    for i in range(10):
        h5f = h5py.File('datasets/lfw_1/lfw_augment_normal_cv_' + str(i) + '.h5', 'w')
        
        X_train = np.concatenate(tuple([X[k] for k in range(10) if k != i]))
        y_train = np.concatenate(tuple([y[k] for k in range(10) if k != i]))
        X_test = X[i]
        y_test = y[i]
        
        h5f.create_dataset('X_train', ((n_scaling+1)*X_train.shape[0], 2, rows, cols, nchannels), dtype='uint8')
        h5f.create_dataset('y_train', ((n_scaling+1)*y_train.shape[0],), dtype='uint8')
        
        # Lets transform the training data
        for idx in range(X_train.shape[0]):
            i = (n_scaling+1)*idx
            print(idx)
            img1 = X_train[idx,0]
            img2 = X_train[idx,1]
            target = y_train[idx]
            
            # Save original images
            h5f['X_train'][i,0] = img1
            h5f['X_train'][i,1] = img2
            h5f['y_train'][i] = target
            
            aug_count = 1
            for aug in augmenters:
                h5f['X_train'][i+aug_count,0] = aug.augment_image(img1)
                h5f['X_train'][i+aug_count,1] = aug.augment_image(img2)
                h5f['y_train'][i+aug_count] = target
                aug_count += 1
        
        
        # Save test set
        h5f.create_dataset('X_test', data = X_test)
        h5f.create_dataset('y_test', data = y_test)
        h5f.close()
    #%%
    n_scaling = 5
    # Parameters from theta estimation
    if not f_exist('params.pkl'):
        set_params()
    params = load_obj('params')
    
    # Create transformation object
    data = Bunch()
    data.kind = 'landmarks'
    data.landmarks_are_lin_ordered = 0
    data.src = CpuGpuArray(np.zeros((params['num_landmarks'], 2)))
    data.dst = CpuGpuArray(np.zeros((params['num_landmarks'], 2)))
    tf = TransformationFitter(nRows=params['imsize'][0], nCols=params['imsize'][1],
                              vol_preserve=params['vol_preserve'],
    						sigma_lm=params['sigma_lm'],
						base=params['base'],
						nLevels=params['nLevels'],
        					valid_outside=params['valid_outside'],
						tess=params['tess'],
						scale_spatial=params['scale_spatial'],
						scale_value=params['scale_value'],
						scale_quiver=params['scale_quiver'],
						zero_v_across_bdry=params['zero_v_across_bdry'])
    tf.set_dense()
    tf.set_data(data)
    tf.set_run_lengths([params['MCMCniters_per_level']] * tf.nLevels)
    
    # Prepare for image warping
    tw = tf.tw    
    
    # Cluster parameters
    m_k, W_k, v_k, Nk, r_nk = load_obj('cluster_data/cluster_parameters_processed')
    K = len(Nk) # number of clusters
    pri = Nk / sum(Nk)
    
    for i in range(10):
        h5f = h5py.File('datasets/lfw_2/lfw_augment_cpab_cv_' + str(i) + '.h5', 'w')
        
        X_train = np.concatenate(tuple([X[k] for k in range(10) if k != i]))
        y_train = np.concatenate(tuple([y[k] for k in range(10) if k != i]))
        X_test = X[i]
        y_test = y[i]
        
        h5f.create_dataset('X_train', ((n_scaling+1)*X_train.shape[0], 2, rows, cols, nchannels), dtype='uint8')
        h5f.create_dataset('y_train', ((n_scaling+1)*y_train.shape[0],), dtype='uint8')
        
        # Lets transform the training data
        for idx in range(X_train.shape[0]):
            i = (n_scaling+1)*idx
            print(idx)
            img1 = X_train[idx,0]
            img2 = X_train[idx,1]
            target = y_train[idx]
            
            # Save original images
            h5f['X_train'][i,0] = img1
            h5f['X_train'][i,1] = img2
            h5f['y_train'][i] = target
            
            for aug_count in range(1, n_scaling+1):   
                cluster_idx = np.random.choice(range(K), 1, replace=False, p = pri)[0]
                theta = np.random.multivariate_normal(m_k[cluster_idx].flatten().astype('float64'), 
                                    np.linalg.pinv(v_k[cluster_idx] * W_k[cluster_idx].astype('float64')))
                h5f['X_train'][i+aug_count,0] = generate_new_images(img1, tw, theta, 1, [1,1])[0]
            
                cluster_idx = np.random.choice(range(K), 1, replace=False, p = pri)[0]
                theta = np.random.multivariate_normal(m_k[cluster_idx].flatten().astype('float64'), 
                                    np.linalg.pinv(v_k[cluster_idx] * W_k[cluster_idx].astype('float64')))
                h5f['X_train'][i+aug_count,1] = generate_new_images(img2, tw, theta, 1, [1,1])[0]
            
                h5f['y_train'][i+aug_count] = target
        
        # Save test set
        h5f.create_dataset('X_test', data = X_test)
        h5f.create_dataset('y_test', data = y_test)
        h5f.close()
           