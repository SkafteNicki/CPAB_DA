#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 11:04:47 2017

@author: nicki
"""

#%% Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

#%% Load data
experiments = np.load('experiments.npy')
experiments = experiments[()]

train_no = experiments['train_no']
train_normal = experiments['train_normal']
train_cpab5 = experiments['train_cpab5']
train_cpab10 = experiments['train_cpab10']
val_no = experiments['val_no']
val_normal = experiments['val_normal']
val_cpab5 = experiments['val_cpab5']
val_cpab10 = experiments['val_cpab10']
test_no = experiments['test_no']
test_normal = experiments['test_normal']
test_cpab5 = experiments['test_cpab5']
test_cpab10 = experiments['test_cpab10']

#%% Plot training
plt.figure()
plt.plot(np.mean(train_no,1))
plt.plot(np.mean(train_normal,1))
plt.plot(np.mean(train_cpab5,1))
plt.plot(np.mean(train_cpab10,1))
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Training accuracy', fontsize=20)
plt.legend(['No augment', 'Normal augment', '5 cpab augment', '10 cpab augment'],
           loc='best')
plt.axis([0,400,0.2,1])

plt.figure()
plt.plot(medfilt(np.mean(train_no,1),7))
plt.plot(medfilt(np.mean(train_normal,1),7))
plt.plot(medfilt(np.mean(train_cpab5,1),7))
plt.plot(medfilt(np.mean(train_cpab10,1),7))
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Training accuracy (smoothed)', fontsize=20)
plt.legend(['No augment', 'Normal augment', '5 cpab augment', '10 cpab augment'],
           loc='best')
plt.axis([0,400,0.2,1])

#%% Plot validation
plt.figure()
plt.plot(np.mean(val_no,1))
plt.plot(np.mean(val_normal,1))
plt.plot(np.mean(val_cpab5,1))
plt.plot(np.mean(val_cpab10,1))
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Validation accuracy', fontsize=20)
plt.legend(['No augment', 'Normal augment', '5 cpab augment', '10 cpab augment'],
           loc='best')
plt.axis([0,400,0.4,0.9])

plt.figure()
plt.plot(medfilt(np.mean(val_no,1), 7))
plt.plot(medfilt(np.mean(val_normal,1), 7))
plt.plot(medfilt(np.mean(val_cpab5,1), 7))
plt.plot(medfilt(np.mean(val_cpab10,1), 7))
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.title('Validation accuracy (smoothed)', fontsize=20)
plt.legend(['No augment', 'Normal augment', '5 cpab augment', '10 cpab augment'],
           loc='best')
plt.axis([0,400,0.4,0.9])

#%% Test results
print('Test results:')
print('   no augmentation           ', np.mean(test_no), '+-', np.std(test_no))
print('   normal augmentation       ', np.mean(test_normal), '+-', np.std(test_normal))
print('   size 5 cpab augmentation  ', np.mean(test_cpab5), '+-', np.std(test_cpab5))
print('   size 10 cpab augmentation ', np.mean(test_cpab10), '+-', np.std(test_cpab10))
print('\n')

#%% Conclusions
# * using 5 normal augmentation samples increase accuracy compared to not using
#   any augmentation scheme -> as expected
# * using 5 cpab augmentated samples compared to 5 normal augmentation samples,
#   cpab wins the race by 2% points
# * using 10 cpab augmentated samples compared to only 5, give a small increase
#   in accuracy, however we also introduce more variance into the model -> higher
#   probability of sampling bad samples?
# * Trying 20 cpab augmentation -> initial conclusion: the accuracy does not seems
#   to increase further -> all variance in dataset seems to be captured using only
#   10 samples.
# * Variance higher on cpab10 than on other algorithms on lfw webpage
# * Maybe try to redo experiments with more epochs -> chance that some of the variance
#   can be reduced
# * Training curves shows that we are still overfitting, that we augmented samples
#   are regulizing the networks since the training accuracy decrease. Still, it
#   seems like we are overfitting -> maybe turn up L2 regulization a bit to increase
#   accuracy?

