import gstools as gs
from gstools import krige
import numpy as np

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
        lx = np.array([np.random.randint(500, 3000), np.random.randint(250, 1500)])
        ang = np.random.uniform(0, 2 * np.pi)
        sigma = np.random.uniform(1, 5)
        if lx[0] < lx[1]:
            lx = np.flip(lx)
            if ang > np.pi:
                ang -= np.pi/2
            else:
                ang += np.pi/2
    elif covtype == 'good':
        lx = clx[0]
        ang = angles[0]
    
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
    
    return np.exp(field[0]),  model, [M[0,0], M[1,0], M[1,1]] #, [lx, ang]
    
        


