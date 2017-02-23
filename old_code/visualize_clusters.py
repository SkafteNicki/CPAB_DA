# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 07:44:13 2016

@author: nicki
"""
import numpy as np
import scipy.io as sio
import math
#import imageio
from utils_func import load_obj, load_lfw, generate_new_images
import matplotlib.pyplot as plt
#import matplotlib as mpl
from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpa2d.inference.transformation.TransformationFitter import TransformationFitter
       
#%%
if __name__ == "__main__":
    ## Parameters for solver
    params = load_obj('params')
    num_lm = params['num_landmarks']

    ## Read data
    thetadata, info, person = load_obj('cluster_data/theta_data')
    
    ## Read LFW data
    (imgs, index, lm, _) = load_lfw()

    ## Create transformation object
    data = Bunch()
    data.kind = 'landmarks'
    data.landmarks_are_lin_ordered = 0
    data.src = CpuGpuArray(np.zeros((num_lm, 2)))
    data.dst = CpuGpuArray(np.zeros((num_lm, 2)))
    tf = TransformationFitter(  nRows=params['imsize'][0], nCols=params['imsize'][1],
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

    ## Prepare for image warping
    tw = tf.tw

    ## Loading parameters from cluster analysis    
    #vardata = load_obj('cluster_data/cluster_parameters_processed')
    m_k, W_k, v_k, Nk, r_nk = load_obj('cluster_data/cluster_parameters_processed')
    K = len(Nk)
    membship = np.argmax(r_nk, axis=1)    
    
    ## Read checkerboard image and landmarks
    checkerboard = sio.loadmat('checkerboard.mat')
        
    new_images = [ ]
    cluster_idx = [ ]
    for k in range(K):
        if Nk[k] / Nk.sum() < 0.01:
            continue
        print k
        
        (eigVal, eigVec) = np.linalg.eig(np.linalg.pinv(v_k[k] * W_k[k].astype('float64')))
        
        img = np.array(checkerboard['image'], order='C')
        landmarks = checkerboard['corners']
        landmarks = np.reshape(landmarks, (300, 2), order='C').copy()
        data.src = CpuGpuArray(landmarks)
#        
        new_img, new_lm = generate_new_images(img, data.src, tw, eigVec[:,0], 1, 2* [math.sqrt(eigVal[0])])
        plt.imsave(arr=new_img[0], fname = 'cluster_directions_' + str(k) + '.png', cmap =plt.cm.gray)      
#        

        # Choose random image from this cluster
        r = np.random.choice(range(0,len(imgs)))
        img = imgs[r]
        landmarks = np.reshape(lm[r], (num_lm, 2), order='C').copy()
        data.src = CpuGpuArray(landmarks)
        theta = np.random.multivariate_normal(m_k[k].flatten().astype('float64'),
                    np.linalg.pinv(v_k[k]* W_k[k].astype('float64')))
        theta = thetadata[np.random.choice(range(0, thetadata.shape[0]))]
        new_img, new_lm = generate_new_images(img, data.src, tw, theta, 1, 2* [1])
        plt.imsave(arr=new_img[0], fname = 'sample_' + str(k) + '.png')      
        
        
#        np_person = 10
#        idx = np.random.choice(np.where(membship == k)[0], np_person, replace = False)        
#        #theta = m_k[k].T.flatten().astype('float64')
#        for i in idx:
#            theta = np.random.multivariate_normal(m_k[k].flatten().astype('float64'), 
#                                                  np.linalg.pinv(v_k[k] * W_k[k].astype('float64')))
#            img = imgs[info[i,2]]
#            landmarks = lm[info[i,2]]
#            landmarks = np.reshape(landmarks, (num_lm,2), order='C').copy()
#            data.src = CpuGpuArray(landmarks)
#            new_img, new_lm = generate_new_images(img, data.src, tw, theta,
#                                                  params['num_new_imgs'], params['theta_gen_scale'])
#            new_images.append(new_img)

#        np_samp = 5
#        np_person = 5
#        idx = np.random.choice(np.where(membship == k)[0], np_person, replace=False, \
#                p = r_nk[np.where(membship == k)[0],k] / sum(r_nk[np.where(membship == k)[0],k]))
#        for i in idx:
#            for j in range(np_samp):
#                theta = np.random.multivariate_normal(m_k[k].flatten().astype('float64'), \
#                        np.linalg.pinv(v_k[k] * W_k[k].astype('float64')))
#                img = imgs[info[i,2]]
#                landmarks = lm[info[i,2]]
#                landmarks = np.reshape(landmarks, (num_lm,2), order='C').copy()
#                data.src = CpuGpuArray(landmarks)
#                new_img, new_lm = generate_new_images(img, data.src, tw, theta, \
#                        params['num_new_imgs'], params['theta_gen_scale'])
#                new_images.append(new_img)
        
#    ## Write to html file
#    f = open('sampler.html', 'wt')
#    f.write('<html><body>\n')        
#    
#    directory = 'gifs'
#    if not os.path.exists(directory):
#        os.makedirs(directory)
#    
#    for cluster in cluster_idx:
#        f.write('\n\n<h1>' + str(cluster) + '</h1>\n<p>')
#        
#        for i in range(np_person):
#            for j in range(np_samp):
#                gif_name = 'cluster_{}_{}_{}.gif'.format(cluster,i,j)
#                imageio.mimsave(directory + '/' + gif_name, np.array(new_images[j+i*np_person]))
#                f.write('  <img src="' + directory + '/' + gif_name + '">\n')
#            f.write('\n')
#    counter = 0
#    for cluster in cluster_idx:
#        f.write('\n\n<h1>' + str(cluster) + '</h1>\n<p>')
#        for i in range(np_person):
#            gif_name = 'cluster_{}_{}.gif'.format(cluster,i)
#            imageio.mimsave(directory + '/' + gif_name, np.array(new_images[counter*np_person+i]))
#            f.write('  <img src="' + directory + '/' + gif_name + '">\n')
#        counter += 1
#    f.write('</body></html>\n')
#    f.close()
#        