# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 17:30:49 2016

@author: Nicki
"""
import numpy as np
from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpa2d.inference.transformation.TransformationFitter import TransformationFitter
import cv2


#%% Function for saving to a pkl file
def save_obj(obj, name):
    import cPickle as pkl
    with open(name + '.pkl', 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)
    
#%% Function for loading a pkl file
def load_obj(name):
    import cPickle as pkl
    with open(name, 'rb') as f:
        return pkl.load(f)

#%% Function for reading alignment data
def load_alignment_data(folder_data, folder_info):
    import os
    import numpy as np    
    
    datadict = dict()
    infodict = dict()

    ## Read .pkl into two dictonaries
    for file in os.listdir(folder_data + '/'):
        if file.endswith('.pkl'):
            mydict1 = load_obj(folder_data + '/' + file)
        datadict.update(mydict1)

    for file in os.listdir(folder_info + '/'):
        if file.endswith('.pkl'):
            mydict2 = load_obj(folder_info + '/' + file)
        infodict.update(mydict2)

    ## Append all theta values into one data matrix (N x d)
    data = np.empty((0,58))
    info = np.empty((0,4))
    person = [ ]
    for p, theta in datadict.items():
        data = np.append(data, theta, axis = 0)
        info = np.append(info, infodict[p], axis = 0)
        person = person + theta.shape[0]*[p]
    return data, info.astype('int'), person

#%% Function for loading lfw dataset
def load_lfw(num_lm):
    import deeppy as dp
    dataset = dp.dataset.LFW('original')
    imgs = dataset.imgs
    landmarks = dataset.landmarks(num_lm)
    index = dataset.index
    attribute_names = dataset.attribute_names
    attributes = dataset.attributes
    return imgs, index, landmarks, attribute_names, attributes

#%% Function for generating new images
def generate_new_images(img, landmarks, tw, theta, num_new_imgs, theta_gen_scale):
    pts_src = tw.pts_src_dense
    pts_inv = CpuGpuArray.zeros_like(pts_src)

    cpa_space = tw.ms.L_cpa_space[-1]
    img_src = CpuGpuArray(img)
    img_fwd = CpuGpuArray.zeros_like(img_src)
    lm_fwd  = CpuGpuArray.zeros_like(landmarks)
    
    ## Compute pixel mapping
    t = np.linspace(theta_gen_scale[0], theta_gen_scale[1], num_new_imgs)
    new_images = [img]
    new_landmarks = [landmarks]
    for k in range(1, num_new_imgs):
        ## Set theta
        theta_k = t[k] * theta
        cpa_space.theta2Avees(theta=theta_k)
        cpa_space.update_pat()
        tw.update_pat_from_Avees(level=-1)
	
        ## Transform image
        tw.calc_T_inv(pts_src, pts_inv, level=-1, int_quality=1)
        tw.remap_fwd_opencv(pts_inv, img_src, img_fwd, interp_method=cv2.INTER_LANCZOS4)
        img_fwd.gpu2cpu()
        new_images.append(img_fwd.cpu.copy())

        ## Transform landmarks
        tw.calc_T_fwd(landmarks, lm_fwd, level=-1)
        lm_fwd.gpu2cpu()
        new_landmarks.append(lm_fwd.cpu.copy())

    return new_images, new_landmarks

#%% Set parameters function
def set_params():
    params = { 
        'num_new_imgs':         10,				# number of new images to generate
        'theta_gen_scale':      [0.0,1.0],		# scaling of theta, new images are generated
        'num_landmarks':        68,				# number of landmarks, 68 or ???
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
    return params        
