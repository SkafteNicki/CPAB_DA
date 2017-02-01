#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 13:49:18 2017

@author: nicki
"""

#%% Packages to import
import os
try:
	import cPickle as pkl
except:
	import pickle as pkl
import random
import sys
import contextlib
import numpy as np
from of.gpu import CpuGpuArray
import cv2

#%% Parameter settings for generating transformations
def set_params():
    params = { 
        'num_new_imgs':         10,	# number of new images to generate
        'theta_gen_scale':      [0.0,1.0],	# scaling of theta, new images are generated
        'num_landmarks':        68,	# number of landmarks, 68 or ???
        'imsize':               [250, 250],
        'vol_preserve':         False,
        'sigma_lm':             10,
        'base':                 [4, 4], # [8, 8]
        'nLevels':              1,
        'valid_outside':        True, # False
        'tess':                 'I',
        'scale_spatial':        1,
        'scale_value':          100, #1000*5,
        'scale_quiver':         1000,
        'zero_v_across_bdry':   [0, 0], # [1, 1] -> then valid_outside needs to be False
        'MCMCniters_per_level': 10000,
        'use_prior':            True,
        'proposal_scale':       0.01,
        'use_local':            True,
        'max_pairs':            100,
    }
    save_obj(params, 'params')

#%% Function to check if a given file exist
def f_exist(name):
    return os.path.isfile(name)

#%% Function for creating a folder if it does not exist
def folder_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

#%% Function for saving to a pkl file
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
    
#%% Function for loading a pkl file
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pkl.load(f)

#%% Function for creating random pairs of images
def random_pairs(min_idx, max_idx, nb_pairs):
    pairs = set()
    while len(pairs) < nb_pairs:
        p1, p2 = random.randint(min_idx, max_idx), random.randint(min_idx, max_idx)
        if p1 != p2 and ((p1,p2) not in pairs) and ((p2,p1) not in pairs):
            pairs.add((p1,p2))
    return pairs
    
#%% Function for supress output
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

#%% Function for loading the lfw data
def load_lfw():
    dataset = np.load('datasets/lfw_original/lfw_original.npz')
    landmarks = np.load('datasets/lfw_original/landmarks.npz')
    
    imgs = dataset['imgs']
    index = dataset['index'][()]
    landmarks = landmarks['landmarks68']
    pair_splits = dataset['pair_splits'][()]
    
    return imgs, index, landmarks, pair_splits

#%% Function for creating a single file with all transformations
def concat_alignment_data(folder_data, folder_info):
    datadict = dict()
    infodict = dict()

    # Read .pkl into two dictonaries
    for file in os.listdir(folder_data + '/'):
        if file.endswith('.pkl'):
            mydict1 = load_obj(folder_data + '/' + os.path.splitext(file)[0])
        datadict.update(mydict1)

    for file in os.listdir(folder_info + '/'):
        if file.endswith('.pkl'):
            mydict2 = load_obj(folder_info + '/' + os.path.splitext(file)[0])
        infodict.update(mydict2)

    ## Append all theta values into one data matrix (N x d)
    data = np.empty((0,58))
    info = np.empty((0,4))
    person = [ ]
    for p, theta in datadict.items():
        data = np.append(data, theta, axis = 0)
        info = np.append(info, infodict[p], axis = 0)
        person = person + theta.shape[0]*[p]
    
    # Save data to a new file
    save_obj([data, info.astype('int'), person], 'cluster_data/theta_data')

#%%
def generate_new_images(img, tw, theta, num_new_imgs, theta_gen_scale):
    pts_src = tw.pts_src_dense
    pts_inv = CpuGpuArray.zeros_like(pts_src)

    cpa_space = tw.ms.L_cpa_space[-1]
    img_src = CpuGpuArray(img)
    img_fwd = CpuGpuArray.zeros_like(img_src)
    
    ## Compute pixel mapping
    theta_span = np.linspace(theta_gen_scale[0], theta_gen_scale[1], num_new_imgs)
    new_images = [ ]
    new_landmarks = [ ]
    for t in theta_span:
        ## Set theta
        theta_k = t * theta
        cpa_space.theta2Avees(theta=theta_k)
        cpa_space.update_pat()
        tw.update_pat_from_Avees(level=-1)
	
        ## Transform image
        tw.calc_T_inv(pts_src, pts_inv, level=-1, int_quality=1)
        tw.remap_fwd_opencv(pts_inv, img_src, img_fwd, interp_method=cv2.INTER_LANCZOS4)
        img_fwd.gpu2cpu()
        new_images.append(img_fwd.cpu.copy())

    return new_images
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
