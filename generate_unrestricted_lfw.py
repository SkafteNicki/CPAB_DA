#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 17:04:13 2017

@author: nicki
"""

#%% Packages
from utils_func import load_lfw, zoom_img, progressBar
import numpy as np
import h5py

#%%
def generate_unrestricted_lfw_datasets(imgs, index, pairs, N_pos=5000):
    for i in range(1): # for the 10 datasets
        try:
            h5f = h5py.File('datasets/lfw_unrestricted/lfw_' + str(i) + '.h5', 'w')
            
            # Create datasets
            X_train = h5f.create_dataset('X_train', shape = (2*N_pos, 2, 125, 125, 3), dtype = np.uint8)
            y_train = h5f.create_dataset('y_train', shape = (2*N_pos, ), dtype = np.uint8)
            X_test = h5f.create_dataset('X_test', shape = (600, 2, 125, 125, 3), dtype = np.uint8)
            y_test = h5f.create_dataset('y_test', shape = (600, ), dtype = np.uint8)
        
            # Write test set to file
            for n in range(600):
                progressBar(n, 600, name='Write test set')
                X_test[n,0] = zoom_img(imgs[index[pairs['test'][i][n][0]][pairs['test'][i][n][1]-1]])
                X_test[n,1] = zoom_img(imgs[index[pairs['test'][i][n][2]][pairs['test'][i][n][3]-1]])
                y_test[n]   = 1 if pairs['test'][i][n][0] == pairs['test'][i][n][2] else 0
            print('\n')
            
            # Get list of persons and their images we can sample from
            sample_list = dict()
            already_taken = set()
            for s in range(10):
                if s != i:
                    for n in range(600):
                        if pairs['test'][s][n][0] not in sample_list.keys():
                            sample_list[pairs['test'][s][n][0]] = [pairs['test'][s][n][1]]
                        elif pairs['test'][s][n][1] not in sample_list[pairs['test'][s][n][0]]:
                            sample_list[pairs['test'][s][n][0]].append(pairs['test'][s][n][1])
                        if pairs['test'][s][n][0] not in sample_list.keys():
                            sample_list[pairs['test'][s][n][0]] = [pairs['test'][s][n][1]]
                        elif pairs['test'][s][n][1] not in sample_list[pairs['test'][s][n][0]]:
                            sample_list[pairs['test'][s][n][0]].append(pairs['test'][s][n][1])
                            
            # Sample positive samples
            counter = 0
            for name in sample_list.keys():
                progressBar(counter, N_pos, name = 'Sample positive')
                if len(sample_list[name]) < 2:
                    continue
                img_index = sample_list[name]
                combinations = ((i,j) for i in img_index for j in img_index if i != j and i < j)
                for idx1, idx2 in combinations:
                    #X_train[counter, 0] = zoom_img(imgs[index[name][idx1-1]])
                    #X_train[counter, 1] = zoom_img(imgs[index[name][idx2-1]])
                    #y_train[counter] = 1
                    counter += 1
#            
#            # Sample negative samples
#            while counter < 2*N_pos:
#                progressBar(counter, 2*N_pos, name = 'Sample negative')
#                sample1, sample2 = random.sample(sample_list, 2)
#                if sample1[0] == sample2[0]: 
#                    continue # not a negative sample, continue
#                if (sample1[0], sample1[1], sample2[0], sample2[1]) not in already_taken and \
#                       (sample1[0], sample2[1], sample2[0], sample1[1]) not in already_taken:
#                    X_train[counter,0] = zoom_img(imgs[index[sample1[0]][sample1[1]-1]])
#                    X_train[counter,0] = zoom_img(imgs[index[sample2[0]][sample2[1]-1]])
#                    y_train[counter] = 1
#                    counter += 1
#                    already_taken.add((sample1[0], sample1[1], sample2[0], sample2[1]))
#            print('\n')
        finally:
            h5f.close()
#%%
if __name__ == '__main__':
    # Load lfw data
    imgs, index, _, pairs = load_lfw()
    
    # Create unrestricted datasets
    generate_unrestricted_lfw_datasets(imgs, index, pairs, N_pos=500)
    
    