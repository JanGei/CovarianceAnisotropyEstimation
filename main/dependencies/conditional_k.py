import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from randomK_points import randomK_points
from covarmat_s import covarmat_s
import numpy as np
# from dependencies.plotting.plot_k_fields import plot_k_fields


def conditional_k(gwf, pars: dict, pp_xy = [], pp_cid = [], covtype = 'random', valtype = 'good', test_cov = []):
    
    cov     = pars['cov']
    clx     = pars['lx']
    angles  = pars['ang']
    sigma   = pars['sigma'][0]
    mu      = pars['mu'][0]
    cov     = pars['cov']
    k_ref   = np.loadtxt(pars['k_r_d'], delimiter = ',')
    
    mg = gwf.modelgrid
    xyz = mg.xyzcellcenters
    cxy = np.vstack((xyz[0], xyz[1])).T
    dxmin = np.min([max(sublist) - min(sublist) for sublist in mg.xvertices])
    dymin = np.min([max(sublist) - min(sublist) for sublist in mg.yvertices])
    dx = [dxmin, dymin]
    sig_meas = 0.1 # standard deviation of measurement error
    
    if covtype == 'random':
        lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]*2),
                       np.random.randint(pars['dx'][1], clx[0][1]*2)])
        ang = np.random.uniform(0, np.pi)
        sigma = np.random.uniform(0.5, 3)
        if lx[0] < lx[1]:
            lx = np.flip(lx)
            if ang > np.pi/2:
                ang -= np.pi/2
            else:
                ang += np.pi/2
        elif lx[0] == lx[1]:
            lx[0] += 1
    elif covtype == 'good':
        lx = clx[0]
        ang = np.deg2rad(angles[0])
    elif covtype == 'test':
        lx = test_cov[0]
        ang = test_cov[1]
        
    # starting k values at pilot points
    if valtype == 'good':
        pp_k = np.log(k_ref[pp_cid.astype(int)])
        # Kg = k_ref_m
        
    elif valtype == 'random':
        pp_k = np.random.uniform(mu-mu/4, mu+mu/4, len(pp_cid))
        pp_k = pp_k + sig_meas * np.random.randn(*pp_k.shape)
        # Kg = np.random.uniform(k_ref_m - k_ref_m/4, k_ref_m + k_ref_m/4)
    
    # random, unconditional field for the given variogram
    Kflat = np.log(randomK_points(mg.extent, cxy, dx,  lx, ang, sigma, cov, 1, pars)) 
    
    
    # Construct covariance matrix of measurement error
    m = len(pp_k)
    n = cxy.shape[0]
    # Discretized trend functions (for constant mean)
    X = np.ones((n,1))
    Xm = np.ones((m,1)) 
    # One = np.ones((1,n))
    
    R = np.eye(m)* sig_meas**2
    
    Ctype = 2
    # Construct the necessary covariance matrices
    Qssm = covarmat_s(cxy,pp_xy,Ctype,[sigma,lx,ang]) 
    Qsmsm = covarmat_s(pp_xy,pp_xy,Ctype,[sigma,lx, ang])
        
    # kriging matrix and its inverse
    krigmat = np.vstack((np.hstack((Qsmsm+R, Xm)), np.append(Xm.T, 0)))
    # ikrigmat = np.linalg.inv(krigmat)
    
    # generating a conditional realisation
    sunc_at_meas = np.zeros(m)
    for ii in range(m):
        sunc_at_meas[ii] = Kflat[int(pp_cid[ii])] 
    
    # Perturb the measurements and subtract the unconditional realization
    spert = np.squeeze(pp_k) + np.squeeze(sig_meas * np.random.randn(*pp_k.shape)) - np.squeeze(sunc_at_meas)
    
    # Solve the kriging equation
    sol = np.linalg.lstsq(krigmat, np.append(spert.flatten(), 0), rcond=None)[0]
    
    # Separate the trend coefficient(s) from the weights of the covariance-functions in the function-estimate form
    xi = sol[:m]
    beta = sol[m]
    
    s_cond = np.squeeze(Qssm.dot(xi)) + np.squeeze(X.dot(beta)) + Kflat
    # plot_k_fields(gwf, pars, [np.exp(Kflat)/1000, np.exp(s_cond)])
    
    #!!!!!!!!!!!! same rotational matrix?
    D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
    return np.exp(s_cond), [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang]