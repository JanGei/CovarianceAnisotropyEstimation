import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from randomK_points import randomK_points
from covarmat_s import covarmat_s
import numpy as np


def conditional_k(mg, cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy, pp_cid):

    # random, unconditional field for the given variogram
    Kflat, K = randomK_points(mg.extent, cxy, dx,  lx, ang, sigma, pars)
    Kflat = np.log(Kflat)

    # Construct covariance matrix of measurement error
    m = len(pp_k)
    n = cxy.shape[0]
    # Discretized trend functions (for constant mean)
    X = np.ones((n,1))
    Xm = np.ones((m,1)) 
    # One = np.ones((1,n))
    
    R = np.eye(m)* pars['sig_me']**2
    
    # Construct the necessary covariance matrices
    Qssm = covarmat_s(cxy,pp_xy,pars,[sigma,lx,ang]) 
    Qsmsm = covarmat_s(pp_xy,pp_xy,pars,[sigma,lx, ang])
        
    # kriging matrix and its inverse
    krigmat = np.vstack((np.hstack((Qsmsm+R, Xm)), np.append(Xm.T, 0)))
    # ikrigmat = np.linalg.inv(krigmat)
    
    # generating a conditional realisation
    sunc_at_meas = np.zeros(m)
    for ii in range(m):
        sunc_at_meas[ii] = Kflat[int(pp_cid[ii])] 
    
    # Perturb the measurements and subtract the unconditional realization
    spert = np.squeeze(pp_k) + np.squeeze(pars['sig_me'] * np.random.randn(*pp_k.shape)) - np.squeeze(sunc_at_meas)
    
    # Solve the kriging equation
    sol = np.linalg.lstsq(krigmat, np.append(spert.flatten(), 0), rcond=None)[0]
    
    # Separate the trend coefficient(s) from the weights of the covariance-functions in the function-estimate form
    xi = sol[:m]
    beta = sol[m]
    
    s_cond = np.squeeze(Qssm.dot(xi)) + np.squeeze(X.dot(beta)) + Kflat

    return np.exp(s_cond)