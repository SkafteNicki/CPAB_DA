'''
Created on Tue Nov 22 10:47:09 2016
@author: Nicki

Description:
    This script implements the variational gaussian mixture of gaussians, and 
    the post-processing of the clusters. The main-script will fit the model
    som 
'''

#%% Packages to import
from utils import load_obj, save_obj, f_exist, concat_alignment_data, folder_create
import numpy as np
import numpy.linalg as la
from scipy.special import psi
from scipy.sparse import spdiags
from scipy.stats import wishart    
import math
import matplotlib.pyplot as plt
import argparse

#%% Variational baysian mixture of gaussian
def vargmm(data, K = 10, iterations = 100, alpha0 = 10e-3, beta0 = 10e-3, \
    v0 = 10, tol = 1e-3, forceZeroMean = False, plot = False):

    # Shape of data
    N, D = data.shape
    
    ## Prior parameters
    v0 = max(v0, D) # cannot be lower than the dimensionality of data
    m0 = np.zeros((1, D))
    W0 = np.eye(D) / v0  

    ## Initilize structures
    m_k = K * [np.empty(D)]
    W_k = K * [np.empty((D,D))]
    beta_k = K * [beta0 + N/K]
    alpha_k = K * [alpha0 + N/K]
    v_k = K * [v0 + N/K]
    Sk = K * [np.empty((D,D))]
    xbar = K * [np.empty(D)]
    
    ## Initilize means and precision of each cluster
    for k in range(K):        
        if forceZeroMean:
            # Let m_k be a zero vector
            m_k[k] = np.zeros((1,D))
            # Let W_k be a random precision matrix from the wishart distribution
            W_k[k] = wishart.rvs(v0, W0)
        else:
            # Let m_k be a random data point
            m_k[k] = data[np.random.randint(low=0, high=N),:]        
            # Let W_k be the mean of wishart prior
            W_k[k] = v0*W0
    
    Nkold = K * [0]    
    ## Main loop
    for i in range(iterations):
        ## Varaitional E-step
        ln_rho = np.zeros((N,K))
        Elnpi = psi(alpha_k) - psi(sum(alpha_k))
        ElnL = np.zeros(K)
        for k in range(K):
            delta_k = data - m_k[k] # N x D
            EmuL = D/beta_k[k] + v_k[k]*np.sum(delta_k.dot(W_k[k]) * delta_k, axis = 1).T # N x 1
            (sign , logdetWk) = la.slogdet(W_k[k])
            ElnL[k] = sum(psi(0.5*(v_k[k] - np.array(range(D))))) + D*math.log(2) + logdetWk # 1 x 1
            ln_rho[:,k] = Elnpi[k] + 0.5*ElnL[k] - 0.5*D*math.log(2*math.pi) - 0.5*EmuL # N x 1

        ## Calculate responsability
        rho = np.zeros((N,K))
        for idx in range(N):
            finite = np.isfinite(ln_rho[idx,:])
            rho[idx, finite] = np.exp(ln_rho[idx,finite] - np.max(ln_rho[idx,finite]))
        r_nk = (rho.T * (1.0 / np.sum(rho, axis = 1))).T # N x K
        
        ## Variational M-step
        Nk = np.sum(r_nk, axis=0) # K x 1
        alpha_k = alpha0 + Nk # K x 1
        beta_k = beta0 + Nk # K x 1        
        v_k = v0 + Nk # K x 1
        for k in range(K):        
            rk = r_nk[: , k] / Nk[k] # N x 1
            rk[np.isnan(rk)] = 0 # N x 1
            xbar[k] = rk.dot(data) # 1 x D
            delta_k = data - xbar[k] # N x D
            Sk[k] = delta_k.T.dot(spdiags(rk, 0, N, N) * delta_k) # D x D = N x D * N x N * N x D
            if not forceZeroMean:            
                m_k[k] = (beta0 * m0 + Nk[k] * xbar[k]) / beta_k[k] # 1 x D
            Winv = la.inv(W0) + Nk[k] * Sk[k] + ((beta0*Nk[k]) / (beta0 + Nk[k])) * ((xbar[k] - m0).T * (xbar[k]-m0))
            W_k[k] = la.pinv(Winv)
        
        ## Check if we can stop early (alternative to the variational lower bound)        
        diff = la.norm(Nk - Nkold,2)
        print 'Iteration {} / {}, diff = {}'.format(i+1, iterations,diff)
        if diff < tol:
            break
        Nkold = Nk
        
        if plot:            
           plt.ion()
           plt.clf()
           plt.plot(range(1,1+K), Nk / Nk.sum(), linewidth=5)
           plt.xlabel('Cluster k', fontsize=20)
           plt.ylabel('Nk in % of total observations',fontsize=20)
           plt.pause(0.05)
    
    # Return output    
    return m_k, W_k, v_k, Nk, r_nk, alpha_k, beta_k
#%% Function for truncating cluster with low support
def post_process_parameters(tol, data, m_k, W_k, v_k, Nk, r_nk, alpha_k, beta_k):
    N, D = data.shape    
    K = len(Nk)
    Knew = np.where(Nk / Nk.sum() > tol)[0]

    # Do a extra expectation step, where we only consider well defined clusters    
    ln_rho = np.zeros((N,K))
    Elnpi = psi(alpha_k) - psi(sum(alpha_k))
    ElnL = np.zeros(K) 
    
    for k in range(K):
        if k in Knew:
            delta_k = data - m_k[k] # N x D
            EmuL = D/beta_k[k] + v_k[k]*np.sum(delta_k.dot(W_k[k]) * delta_k, axis = 1).T # N x 1        
            (sign , logdetWk) = la.slogdet(W_k[k])
            ElnL[k] = sum(psi(0.5*(v_k[k] - np.array(range(D))))) + D*math.log(2) + logdetWk # 1 x 1
            ln_rho[:,k] = Elnpi[k] + 0.5*ElnL[k] - 0.5*D*math.log(2*math.pi) - 0.5*EmuL # N x 1
        else:
            ln_rho[:, k] = np.inf
    
    ## Calculate responsability
    rho = np.zeros((N,K))
    for idx in range(N):
        finite = np.isfinite(ln_rho[idx,:])
        rho[idx, finite] = np.exp(ln_rho[idx,finite] - np.max(ln_rho[idx,finite]))
        r_nk = (rho.T * (1.0 / np.sum(rho, axis = 1))).T # N x K            
    r_nk = (rho.T * (1.0 / np.sum(rho, axis = 1))).T # N x K        
    Nk = np.sum(r_nk, axis=0) # K x 1
    
    return m_k, W_k, v_k, Nk, r_nk, alpha_k, beta_k

#%% Variational mixture of gaussians
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
        '''This program fits a variational mixture of gaussians to alignment data
        generated from the "generate_theta.py" script i.e. a folder name "gen_theta"
        and "gen_theta_info" needs to exist with some .ply files in them.''')
    parser.add_argument('-K', action="store", dest="K", type = int, default = 10,
                        help = '''Controls the number of cluster to fit to data''')
    parser.add_argument('-i', action="store", dest="iter", type = int, default = 100,
                        help = '''Controls the maximum number of iterations the 
                        algorithm run''')
    parser.add_argument('-fm', action="store", dest="forcezeromean", type = bool, default = True,
                        help = '''Binary variable that controls if zero-mean variation
                        should be fitted to the data''')
    parser.add_argument('-a0', action="store", dest = 'alpha0', type = float, default = 1e-3,
                        help = '''Prior parameter alpha_0''')
    parser.add_argument('-b0', action="store", dest = 'beta0', type = float, default = 1e-3,
                        help = '''Prior parameter beta_0''')
    parser.add_argument('-v0', action="store", dest = 'v0', type = float, default = 100,
                        help = '''Prior parameter v_0''')
    parser.add_argument('-t', action="store", dest = 'tol', type = float, default = 1e-1,
                        help = '''Tolerance for change between two iterations. If N[k]-N[k-1]
                        if less than the tolerance we stop early''')                
    parser.add_argument('-p', action="store", dest = 'plot', type = bool, default = False,
                        help = '''Control if results are plotted each iterations''')                
    parser.add_argument('-thres', action="store", dest = 'threshold', type = float, default = 0.01,
                        help = '''Threshold parameter for the post-processing. If N[k]/sum(N[k])
                        is smaller than the threshold, the cluster parameter are set to 0''')
    res = parser.parse_args()
    
    print "Running varGMM with settings"
    print "  fitting K cluster:            ", res.K
    print "  in number of iterations:      ", res.iter
    print "  with priors alpha0, beta0, v0 ", res.alpha0, res.beta0, res.v0
    print "  with zeromean:                ", res.forcezeromean
    print "  with tolerance:               ", res.tol
    print "  and plotting:                 ", res.plot
    print "  and post-process threshold    ", res.threshold
    
    # Names for input and output files
    alignment_name = 'cluster_data/theta_data'
    cluster_parameters_name = 'cluster_data/cluster_parameters'
    folder_create('cluster_data')    
    
    folder_theta = 'gen_theta'    
    folder_theta_info = 'gen_theta_info'
    if not f_exist(alignment_name + '.pkl'):
        concat_alignment_data(folder_theta, folder_theta_info)
    thetadata, info, person = load_obj(alignment_name)
    
    # Fit vargmm to data
    m_k, W_k, v_k, Nk, r_nk, alpha_k, beta_k = vargmm(thetadata, K = res.K, 
        iterations = res.iter, alpha0 = res.alpha0, beta0 = res.beta0, 
        v0 = res.v0, tol = res.tol, forceZeroMean = res.forcezeromean, plot = res.plot)
    # Save results
    save_obj([m_k, W_k, v_k, Nk, r_nk], cluster_parameters_name)

    # Post-process parameters
    m_k, W_k, v_k, Nk, r_nk, alpha_k, beta_k = \
        post_process_parameters(res.threshold, thetadata, m_k, W_k, v_k, Nk, r_nk, alpha_k, beta_k)
    
    # Save results
    save_obj([m_k, W_k, v_k, Nk, r_nk], cluster_parameters_name + '_processed')
    