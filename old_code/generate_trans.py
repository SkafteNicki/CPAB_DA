"""
Created on Tue Nov 29 11:07:54 2016
@author: Nicki

Description:
    This script generates a number of presampled transformations from each of the
    cluster generated by the fit_clusters.py script. This is needed to not 
    comflic theano when using the gpu
"""
#%% Packages to use
from utils_func import load_obj, f_exist, set_params, save_obj
import argparse
import numpy as np
from of.utils import *
from of.gpu import CpuGpuArray
from cpab.cpa2d.inference.transformation.TransformationFitter import TransformationFitter

#%% Function for generating transformations
def generate_transformers(n_trans, tw, m_k, v_k, W_k):
    K = len(m_k)
    pts_src = tw.pts_src_dense
    pts_inv = CpuGpuArray.zeros_like(pts_src)
    
    trans = K * [ n_trans * [[]] ]    
    for k in range(K):
        print('Cluster {} / {}'.format(k,K))
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


#%% Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''This program will generate a
            number of transformations such that we do not comflict theano with 
            the gpu''')
    parser.add_argument('-nt', action="store", dest="n_trans", type = int, default = 10000,
                        help = '''Number of transformations generated per cluster''')
    res = parser.parse_args()
    
    n_trans = res.n_trans
    print "Generating cluster transformation with"
    print "  with n_trans per cluster: ", n_trans
                        
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
    
    # Load estimated cluster parameters    
    m_k, W_k, v_k, Nk, r_nk = load_obj('cluster_data/cluster_parameters_processed')
    K = len(Nk) # number of clusters

    # Generate transformations
    generate_transformers(n_trans, tw, m_k, v_k, W_k)