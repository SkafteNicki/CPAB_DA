'''
Created on Tue Nov 22 10:47:09 2016
@author: Nicki

Description:
    This script is responsable for generating the theta values. This is done by
    iterating trough the lfw dataset and estimate transformations between pairs
    of images within each person.    
'''
#%% Packages to import
import argparse
from utils import set_params, f_exist, load_obj, save_obj, load_lfw, random_pairs, nostdout
import numpy as np
from of.gpu import CpuGpuArray
from of.utils import *
from cpab.cpa2d.inference.transformation.TransformationFitter import TransformationFitter
import os

#%% Main function
def augment_lfw(person_range):
    # Parameters for solver
    if not f_exist('params.pkl'):
        set_params()
    params = load_obj('params')
    num_lm = params['num_landmarks']

    # Read data
    (imgs, index, landmarks, _, _, _) = load_lfw(str(num_lm))
    
    # Create transformation object
    data = Bunch()
    data.kind = 'landmarks'
    data.landmarks_are_lin_ordered = 0
    data.src = CpuGpuArray(np.zeros((num_lm, 2)))
    data.dst = CpuGpuArray(np.zeros((num_lm, 2)))
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
	
    # Structures for holding the the data
    count = 0
    person_num = -1
    gen_theta = dict()
    gen_theta_info = dict()
	
    person_nb = 0
    for person, idx in index.items():		
        # Is this a person we want to consider
        person_num += 1
        if person_num < person_range[0] or person_num > person_range[1]:
            continue
		
        # We only do something when we have more than one image pr. person
        N = len(idx)
        print('person nb: ', person_nb, ' N: ', N, 'count: ', count, 'person: ', person)
        person_nb += 1
        
        if N <= 1:
            count += 1
            continue
        
        # Draw random subset of size max_pairs for give
        num_reps = min(params['max_pairs'], (N**2 - N)/2)
        all_pairs = random_pairs(0,N-1,num_reps)	

        # Structures for holding alignment informations
        gen_theta[person] = np.zeros((2*num_reps, tf.tw.ms.L_cpa_space[-1].d)) 
        gen_theta_info[person] = np.zeros((2*num_reps, 4))
        pair_nb = 0
        
        for pair in all_pairs:
            im1, im2 = pair
            
            # Calculate mean and standard deviation for landmarks in image 1
            lm1 = np.reshape(landmarks[idx[im1]], (num_lm, 2), order='F').copy()
            mu1 = np.mean(lm1, axis=0)
            s1 = np.std(lm1)
            data.src = CpuGpuArray(lm1)
            
            # Calculate mean and standard deviation for landmarks in image 2. Standardize with respect to image 1
            lm2 = np.reshape(landmarks[idx[im2]], (num_lm, 2), order='F').copy()
            mu2 = np.mean(lm2, axis=0)
            s2 = np.std(lm2)
            lm2 = (lm2 - mu2) * (s1 / s2) + mu1
            data.dst = CpuGpuArray(lm2)
            
            # Perform fit  of transformation
            tf.set_data(data)
            with nostdout():
                theta, _ = tf.fit(  use_prior=params['use_prior'],
                                    proposal_scale=params['proposal_scale'],
                                    use_local=params['use_local'])
            # Remember all alignments
            gen_theta[person][pair_nb,:] = np.array(theta)
            gen_theta[person][pair_nb+1,:] = np.array(-theta)
            gen_theta_info[person][pair_nb,:] = np.array([im1, im2, idx[im1], idx[im2]])
            gen_theta_info[person][pair_nb+1,:] = np.array([im2, im1, idx[im2], idx[im1]])
            pair_nb += 2
    
    # If we have processed some people save the results
    if not len(gen_theta.keys()) == 0:
        folder_theta = 'gen_theta'
        folder_create(folder_theta)
            
        folder_theta_info = 'gen_theta_info
        folder_create(folder_theta_info)        
        
        postfix = '{}_{}'.format(person_range[0], person_range[1])
        save_obj(gen_theta, folder_theta + '/' + 'gen_theta_{}'.format(postfix))
        save_obj(gen_theta_info, folder_theta_info + '/' + 'gen_theta_info_{}'.format(postfix)) 
	
#%% Main script 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=
        '''This program generates transformations parametizations based on pairs
        of images in the lfw dataset. Most parameters should be changed in the
        set_params() function found in the utils.py file''')
    parser.add_argument('-pr', action="store", dest="person_range", nargs = '+',
                        help = '''person range [min, max] i.e which persons to
                        estimate transformations between. Default is all. ''',
                        default = [-1, 20000])
    parser.add_argument('-maxp', action="store", dest="max_pairs", default = 10,
                        help = '''controls the maximum number of pairs for each
                        person. Set to 200.000 to get all possible combinations''')
    res = parser.parse_args()
    person_range = res.person_range
    max_pairs = res.max_pairs    

    print "Process person range   :", person_range
    print " with max pairs set to :", max_pairs

    # Run theta estimatation
    max_person_range = 1000 # for some reason it has problems with processing
                            # to many people, so split up in smaller parts if
                            # this is the case
    if person_range[1] - person_range[0] < max_person_range:        
        augment_lfw(person_range)
    else:
        for i in range(person_range[0], person_range[1], max_person_range):
            augment_lfw([i, i + max_person_range])
    