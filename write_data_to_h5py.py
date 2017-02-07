#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:13:25 2017

@author: nicki
"""

#%%
import h5py
from utils_func import load_lfw, create_img_pairs, generate_new_images, load_obj, f_exist, set_params
import imgaug.imgaug.augmenters as iaa
import numpy as np
from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpa2d.inference.transformation.TransformationFitter import TransformationFitter

#%%
if __name__ == '__main__':
    imgs, index, landmarks, pairs = load_lfw()
    
    X_train, y_train, X_val, y_val, X_test, y_test = create_img_pairs(imgs, index, pairs)
    
    #%%
    h5f = h5py.File('lfw_augment_no.h5', 'w')
    h5f.create_dataset('X_train', data = X_train)
    h5f.create_dataset('y_train', data = y_train)
    h5f.create_dataset('X_val', data = X_val)
    h5f.create_dataset('y_val', data = y_val)
    h5f.create_dataset('X_test', data = X_test)
    h5f.create_dataset('y_test', data = y_test)
    h5f.close()
    
    #%%
    n_scaling = 5
    h5f = h5py.File('lfw_augment_normal.h5', 'w')
    h5f.create_dataset('X_train', ((n_scaling+1)*X_train.shape[0], 2, 250, 250, 3), dtype='uint8')
    h5f.create_dataset('y_train', ((n_scaling+1)*y_train.shape[0],), dtype='uint8')
    h5f.create_dataset('X_val', data = X_val)
    h5f.create_dataset('y_val', data = y_val)
    h5f.create_dataset('X_test', data = X_test)
    h5f.create_dataset('y_test', data = y_test)
    # Do stuff for X_train and y_train
    augmenters = [ iaa.Fliplr(0.5),
                   iaa.Affine(rotate = (-45, 45)),
                   iaa.GaussianBlur(sigma=(0, 5.0)),
                   iaa.Crop(percent=(0, 0.1)),
                   iaa.Affine(scale = {"x": (0.8, 1.2), "y": (0.8, 1.2)})]
    assert len(augmenters) == n_scaling
    for idx in range(X_train.shape[0]):
        i = (n_scaling+1)*idx
        # Get images and target
        im1 = X_train[idx,0]
        im2 = X_train[idx,1]
        target = y_train[idx]

        # Save original images
        h5f['X_train'][i,0] = im1
        h5f['X_train'][i,1] = im2
        h5f['y_train'][i] = target
        
        aug_count = 1
        for aug in augmenters:
            h5f['X_train'][i+aug_count,0] = aug.augment_image(im1)
            h5f['X_train'][i+aug_count,1] = aug.augment_image(im2)
            h5f['y_train'][i+aug_count] = target
            aug_count += 1
    
    h5f.close()
    #%%
    n_scaling = 50
    h5f = h5py.File('lfw_augment_cpab.h5', 'w')
    h5f.create_dataset('X_train', ((n_scaling+1)*X_train.shape[0], 2, 250, 250, 3), dtype='uint8')
    h5f.create_dataset('y_train', ((n_scaling+1)*y_train.shape[0], ), dtype='uint8')
    h5f.create_dataset('X_val', data = X_val)
    h5f.create_dataset('y_val', data = y_val)
    h5f.create_dataset('X_test', data = X_test)
    h5f.create_dataset('y_test', data = y_test)
    # Do stuff X_train and y_train
    
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
    m_k, W_k, v_k, Nk, r_nk = load_obj('cluster_data/cluster_parameters')
    K = len(Nk) # number of clusters
    pri = Nk / sum(Nk)
    for idx in range(X_train.shape[0]):
        if idx % 100 == 0: print(idx)
        i = (n_scaling+1)*idx
        # Get images and target
        im1 = X_train[idx,0]
        im2 = X_train[idx,1]
        target = y_train[idx]

        # Save original images
        h5f['X_train'][i,0] = im1
        h5f['X_train'][i,1] = im2
        h5f['y_train'][i] = target
           
        for aug_count in range(1, n_scaling+1):   
            cluster_idx = np.random.choice(range(K), 1, replace=False, p = pri)[0]
            theta = np.random.multivariate_normal(m_k[cluster_idx].flatten().astype('float64'), 
                                    np.linalg.pinv(v_k[cluster_idx] * W_k[cluster_idx].astype('float64')))
            h5f['X_train'][i+aug_count,0] = generate_new_images(im1, tw, theta, 1, [1,1])[0]
            
            cluster_idx = np.random.choice(range(K), 1, replace=False, p = pri)[0]
            theta = np.random.multivariate_normal(m_k[cluster_idx].flatten().astype('float64'), 
                                    np.linalg.pinv(v_k[cluster_idx] * W_k[cluster_idx].astype('float64')))
            h5f['X_train'][i+aug_count,1] = generate_new_images(im2, tw, theta, 1, [1,1])[0]
            
            h5f['y_train'][i+aug_count] = target
    h5f.close()
