"""
Created on Tue Nov 22 11:42:27 2016
@author: Nicki

Description:
    This script contains various functions to support other scrips
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
#import cStringIO
import deeppy as dp
import numpy as np
from of.gpu import CpuGpuArray
import cv2

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

#%% Parameter settings for generating transformations
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
    save_obj(params, 'params')
    
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
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = cStringIO.StringIO()
    yield
    sys.stdout = save_stdout   

#%% Function for loading the lfw data
def load_lfw(num_lm):
    dataset = dp.dataset.LFW('original')
    imgs = dataset.imgs
    landmarks = dataset.landmarks(num_lm)
    index = dataset.index
    attribute_names = dataset.attribute_names
    attributes = dataset.attributes
    splits = dataset.people_splits
    return imgs, index, landmarks, attribute_names, attributes, splits
#%% Function for loading lfw pairs
def load_lfw_pairs():
    dataset = dp.dataset.LFW('original')
    imgs = dataset.imgs
    index = dataset.index
    people_split = dataset.pair_splits
    return imgs, index, people_split

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

#%% Function for generating new images
def generate_new_images(img, landmarks, tw, theta, num_new_imgs, theta_gen_scale):
    pts_src = tw.pts_src_dense
    pts_inv = CpuGpuArray.zeros_like(pts_src)

    cpa_space = tw.ms.L_cpa_space[-1]
    img_src = CpuGpuArray(img)
    img_fwd = CpuGpuArray.zeros_like(img_src)
    lm_fwd  = CpuGpuArray.zeros_like(landmarks)
    
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

        ## Transform landmarks
        tw.calc_T_fwd(landmarks, lm_fwd, level=-1)
        lm_fwd.gpu2cpu()
        new_landmarks.append(lm_fwd.cpu.copy())

    return new_images, new_landmarks

#%% Function that removed images that do not have attributes and return a permuted version
def get_lfw_network_data(splittype, imgs, landmarks, index, attribute_names, attributes, split, theta_persons, Nk, r_nk):
    if splittype == 'train' or splittype == 'val':
        N = sum([len(attributes[p]) if p in attributes else 0 for p in split[splittype]])
        lfw_img = np.zeros(shape=(N, 250, 250, 3), dtype=np.uint8)
        lfw_lm  = np.zeros(shape=(N, landmarks.shape[1]),dtype=np.float)
        lfw_att = np.zeros(shape=(N, len(attribute_names)),dtype=np.uint8)
        lfw_pri = np.zeros(shape=(N, len(Nk)),dtype=np.float)       
        counter = 0
    
    elif splittype == 'test':
        N = [sum([len(attributes[p]) if p in attributes else 0 for p in split[splittype][t]]) for t in range(10)]
        lfw_img = 10 * [[ ]]
        lfw_lm =  10 * [[ ]]
        lfw_att = 10 * [[ ]]
        lfw_pri = 10 * [[ ]]
        for t in range(10):
            lfw_img[t] = np.zeros(shape=(N[t], 250, 250, 3), dtype=np.uint8)
            lfw_lm[t]  = np.zeros(shape=(N[t], landmarks.shape[1]),dtype=np.float)
            lfw_att[t] = np.zeros(shape=(N[t], len(attribute_names)),dtype=np.uint8)
            lfw_pri[t] = np.zeros(shape=(N[t], len(Nk)),dtype=np.float)
            counter = 10*[0]    
        
    for person, idx in index.items():
        if person not in attributes: 
            continue # continue if person do not have attributes
        person_attributes = attributes[person]
        # If a person have not contributed to the cluster estimation, 
        # sample cluster k with properbility proportional to the number
        # of theta values in that cluster
        if person not in np.unique(theta_persons):
            pi_k = Nk * 1.0/ sum(Nk)
            # If a person have contributed with samples, samples cluster with
            # probability proportional to the sum of responsabilitys of that cluster 
        else:
            person_theta_idx = np.where(person == np.array(theta_persons))[0]
            M = len(person_theta_idx)
            pi_k = np.sum(r_nk[person_theta_idx,:], axis=0) * 1.0/ M
       
        for i in range(len(idx)): # for all images for that person
            if i+1 not in person_attributes:
                continue # if this image do not have attributes
            if splittype == 'train' or splittype == 'val':            
                if person in split[splittype]:
                    lfw_img[counter] = imgs[idx[i]]
                    lfw_lm[counter] = landmarks[idx[i]]
                    lfw_att[counter] = (np.sign(person_attributes[i+1])>0).astype('int')
                    lfw_pri[counter] = pi_k
                    counter += 1
            elif splittype == "test":
                for t in range(10):
                    if person in split[splittype][t]:
                        lfw_img[t][counter[t]] = imgs[idx[i]]
                        lfw_lm[t][counter[t]] = landmarks[idx[i]]
                        lfw_att[t][counter[t]] = (np.sign(person_attributes[i+1])>0).astype('int')
                        lfw_pri[t][counter[t]] = pi_k
                        counter[t] += 1
        
    # Permute data
    if splittype == 'train' or splittype == 'val':
        perm = np.random.permutation(N)
        lfw_img = lfw_img[perm]
        lfw_lm = lfw_lm[perm]
        lfw_att = lfw_att[perm]
        lfw_pri = lfw_pri[perm]
    elif splittype == 'test':
        for t in range(10):
            perm = np.random.permutation(N[t])
            lfw_img[t] = lfw_img[t][perm]
            lfw_lm[t] = lfw_lm[t][perm]
            lfw_att[t] = lfw_att[t][perm]
            lfw_pri[t] = lfw_pri[t][perm]
    return lfw_img, lfw_lm, lfw_att, lfw_pri   
#%% Function that returns a augmented version of the batch if wanted
def get_batch_data(n_samples, imgs, landmarks, attributes, priors, m_k, v_k, W_k, tw, data, params):
    if n_samples == 1:
        return np.rollaxis(imgs, 3, 1), attributes
    else:
        K = len(m_k)
        new_images = np.zeros(shape=(n_samples*len(imgs), imgs.shape[3], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
        new_attributes = np.zeros(shape=(n_samples*len(imgs), attributes.shape[1]), dtype=np.uint8)
        count = 0
        for i in range(imgs.shape[0]):

            new_images[count] = np.rollaxis(imgs[i], 2)
            new_attributes[count] = attributes[i]
            count += 1
            
            image = imgs[i]
            lm = np.reshape(landmarks[i], (params['num_landmarks'], 2), order='C').copy()
            data.src = CpuGpuArray(lm)            
            
            for n in range(n_samples-1):
                # Draw random cluster index
                cluster = np.random.choice(range(K), size = 1, p = priors[i])[0]
                # Draw random sample from that cluster
                theta = np.random.multivariate_normal(m_k[cluster].flatten().astype('float64'), 
                      np.linalg.pinv(v_k[cluster] * W_k[cluster].astype('float64')))
                # Generate new sample
                (new_img, _) = generate_new_images(image, data.src, tw, theta, 1, [1, 1])
                new_images[count] = np.rollaxis(new_img[0],2)
                new_attributes[count] = attributes[i]
                count += 1
        
        return new_images, new_attributes

#%% Function for generating transformations ready for apply
def generate_transformers(n_trans, tw, m_k, v_k, W_k):
    K = len(m_k)
    pts_src = tw.pts_src_dense
    pts_inv = CpuGpuArray.zeros_like(pts_src)
    
    trans = K * [ n_trans * [[]] ]    
    for k in range(K):
        for n in range(n_trans):        
            theta = np.random.multivariate_normal(m_k[k].flatten().astype('float64'), 
                                    np.linalg.pinv(v_k[k] * W_k[k].astype('float64')))
        
            cpa_space = tw.ms.L_cpa_space[-1]
            cpa_space.theta2Avees(theta=theta)
            cpa_space.update_pat()
            tw.update_pat_from_Avees(level=-1)
            
            ## Transform object
            tw.calc_T_inv(pts_src, pts_inv, level=-1, int_quality=1)
            pts_inv.gpu2cpu()
            trans[k][n] = pts_inv.cpu.copy()
            
    save_obj(trans, 'transformations')


#%% Function for returning a batch
def get_batch_data2(n_samples, imgs, attributes, priors, trans):
    if n_samples == 1:
        return np.rollaxis(imgs, 3, 1), attributes
    else:
        K = len(trans)
        n_trans = len(trans[0])
        new_images = np.zeros(shape=(n_samples*len(imgs), imgs.shape[3], imgs.shape[1], imgs.shape[2]), dtype=np.uint8)
        new_attributes = np.zeros(shape=(n_samples*len(imgs), attributes.shape[1]), dtype=np.uint8)
        count = 0
        for i in range(imgs.shape[0]):

            new_images[count] = np.rollaxis(imgs[i], 2)
            new_attributes[count] = attributes[i]
            count += 1
            
            image = imgs[i]
            for n in range(n_samples-1):
                # Draw random cluster index
                cluster = np.random.choice(range(K), size = 1, p = priors[i])[0]

                # Draw a random generated transformer
                rand_trans = np.random.randint(0, n_trans)                
                
                # Get random transformation from cluster
                pts_fwd = trans[cluster][rand_trans]
                
                # Do image deformation                
                map1 = pts_fwd[:,0].astype(np.float32).reshape(image.shape[:2])
                map2 = pts_fwd[:,1].astype(np.float32).reshape(image.shape[:2])
                new_img = np.zeros_like(image)
                cv2.remap(src = image, map1 = map1, map2 = map2, interpolation = cv2.INTER_LANCZOS4, dst = new_img)
                new_images[count]=np.rollaxis(new_img,2)
                new_attributes[count] = attributes[i]
                count += 1
				                       
        return new_images, new_attributes
