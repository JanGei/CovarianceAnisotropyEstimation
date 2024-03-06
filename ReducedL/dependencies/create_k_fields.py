import gstools as gs
from gstools import krige
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from randomK import randomK

def create_conditional_k_fields(gwf, pars: dict, pp_xy, pp_cid: np.ndarray, covtype = 'random', valtype = 'good'):
    dim = 2
    cov = pars['cov']
    clx     = pars['lx']
    nx      = pars['nx']
    dx      = pars['dx']
    angles  = pars['ang']
    sigma   = pars['sigma'][0]
    mu      = pars['mu'][0]
    cov     = pars['cov']
    k_ref   = pars['k_ref']

    mg = gwf.modelgrid
    xcell,ycell, _ = mg.xyzcellcenters
    
    if covtype == 'random':
        lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]),
                       np.random.randint(pars['dx'][1], clx[0][1])])
        ang = np.random.uniform(0, 2 * np.pi)
        sigma = np.random.uniform(1, 5)
        if lx[0] > lx[1]:
            lx = np.flip(lx)
            if ang > 0:
                ang -= np.pi/2
            else:
                ang += np.pi/2
        elif lx[0] == lx[1]:
            lx[0] += 1
    elif covtype == 'good':
        lx = clx[0]
        ang = np.deg2rad(angles[0])
        
    # starting k values at pilot points
    if valtype == 'good':
        pp_k = k_ref[pp_cid.astype(int)]
        Kg = np.mean(np.mean(k_ref))
    elif valtype == 'random':
        smeas = k_ref[pp_cid.astype(int)]
        sig_meas = 0.1 # standard deviation of measurement error
        smeas = smeas + sig_meas * np.random.randn(*smeas.shape)
        Kg = np.random.uniform(mu-mu/2, mu+mu/2)
        
    # generating a random field with defined variogram
    K = randomK(nx, dx, lx, ang, sigma, cov, Kg)
    
    # generating a conditional realisation
    
    
    

def create_k_fields(gwf, pars: dict, pp_xy, pp_cid: np.ndarray, covtype = 'random', valtype = 'good'):
    dim = 2
    cov = pars['cov']
    clx     = pars['lx']
    angles  = pars['ang']
    sigma   = pars['sigma'][0]
    mu      = pars['mu'][0]
    cov     = pars['cov']
    k_ref   = pars['k_ref']

    mg = gwf.modelgrid
    xyz = mg.xyzcellcenters
    
    if covtype == 'random':
        lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]),
                       np.random.randint(pars['dx'][1], clx[0][1])])
        ang = np.random.uniform(0, 2 * np.pi)
        sigma = np.random.uniform(1, 5)
        if lx[0] > lx[1]:
            lx = np.flip(lx)
            if ang > 0:
                ang -= np.pi/2
            else:
                ang += np.pi/2
        elif lx[0] == lx[1]:
            lx[0] += 1
    elif covtype == 'good':
        lx = clx[0]
        ang = np.deg2rad(angles[0])
    
    if cov == 'Matern':
        model = gs.Matern(dim=dim, var=sigma, angles = ang, len_scale=lx)
    elif cov == 'Exponential':
        model = gs.Exponential(dim = dim, var = sigma, len_scale=lx, angles = ang)
    elif cov == 'Gaussian':
        model = gs.Gaussian(dim = dim, var = sigma, len_scale=lx, angles = ang)
    
    # starting k values at pilot points
    if valtype == 'good':
        pp_k = k_ref[pp_cid.astype(int)]
    elif valtype == 'random':
        pp_k = np.exp(np.random.randn(len(pp_cid)) + mu)
    
    krig = krige.Ordinary(model, cond_pos=(pp_xy[:,0], pp_xy[:,1]), cond_val = np.log(pp_k))
    field = krig((xyz[0], xyz[1]))

    # convert it to matrix form
    # rotation matrix 
    D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
    # # check angles --- so sollte es immer hinhauen, wenn wir zuerst sortieren
    # eigenvalues, eigenvectors = np.linalg.eig(M) 
    
    # lxmat = 1/np.sqrt(eigenvalues)
    
    # if lxmat[0] < lxmat[1]:
    #     lxmat = np.flip(lxmat)
    #     eigenvectors = np.flip(eigenvectors, axis = 1)
        
    
    # if eigenvectors[0,0] > 0:
    #     ang_test = np.pi/2 -np.arccos(np.dot(eigenvectors[:,0],np.array([0,1])))    
    #     case = 1
    # else:
    #     if eigenvectors[1,0] > 0:
    #         ang_test = np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))
    #         case = 2
    #     else:
    #         ang_test = np.pi -np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))
    #         case = 3
    
            
    
    # if lx[0] < lx[1]:
    #     lx_target = np.flip(lx)
    #     ang_target = (ang+ np.pi/2)%np.pi
    # else: 
    #     lx_target = lx
    #     ang_target = ang
        
    # tolerance = 0.1        
    # if abs(ang_test - ang_target) < tolerance or abs(ang_test - (ang_target - np.pi)) < tolerance or abs(ang_test - (ang_target + np.pi)) < tolerance:
    #     pass
    # else:
    #     print(f'wrong angle in case {case}')
    #     print(ang_test - ang_target)
    
    # if np.round(lxmat[0]) == lx_target[0] and np.round(lxmat[1]) == lx_target[1]:
    #     pass
    #     # print(f'correct l extraction in quadrants {quadrants} with {format(angle, ".2f")}°')
    # else:
    #     print(f'wrong extraction in case {case}')
         
    
    return np.exp(field[0]),  model, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang]
    
        


