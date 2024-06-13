import numpy as np
from dependencies.conditional_k import conditional_k
from dependencies.Kriging import Kriging

def create_k_fields(gwf, pars: dict, pp_xy = [], pp_cid = [], test_cov = [], conditional = True, random = True):
    
    clx     = pars['lx']
    angles  = pars['ang']
    sigma   = pars['sigma'][0]
    k_ref   = np.loadtxt(pars['k_r_d'], delimiter = ',')
    dx      = pars['dx']
    
    mg = gwf.modelgrid
    xyz = mg.xyzcellcenters
    cxy = np.vstack((xyz[0], xyz[1])).T
    sig_meas = pars['sig_me']
    
    if pars['estyp'] == "overestimate":
        factor = 3
    elif pars['estyp'] == "good":
        factor = 2
    elif pars['estyp'] == "underestimate":
        factor = 1
    
    if pars['covt'] == 'random':
        lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]*factor),
                       np.random.randint(pars['dx'][1], clx[0][1]*factor)])
        ang = np.random.uniform(0, np.pi)
        if lx[0] < lx[1]:
            lx = np.flip(lx)
            if ang > np.pi/2:
                ang -= np.pi/2
            else:
                ang += np.pi/2
        elif lx[0] == lx[1]:
            lx[0] += 1
    elif pars['covt'] == 'good':
        lx = clx[0]
        ang = np.deg2rad(angles[0])
    if not random:
        lx = clx[0]
        ang = np.deg2rad(angles[0])
        
    if test_cov:
        lx = test_cov[0]
        ang = test_cov[1]
        
    # starting k values at pilot points
    if pars['valt'] == 'good':
        pp_k = np.log(k_ref[pp_cid.astype(int)]) 
        pp_k = pp_k + sig_meas * np.random.randn(*pp_k.shape)
    elif pars['valt'] == 'random':
        
        # random sampling from mean uniform distribution centered around mean
        mu = pars['mu'][0]
        std = pars['sigma'][0]
        pp_k = np.random.normal(mu, std, len(pp_cid)) #np.random.uniform(mu-mu/4, mu+mu/4, len(pp_cid))
        pp_k = pp_k + sig_meas * np.random.randn(*pp_k.shape)
    
    if conditional:
        field, field2f = conditional_k(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy)
    else:
        field, field2f = Kriging(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy)
    
    # The ellips is rotated counter-clockwise, thats why a minus is needed here
    D = pars['rotmat'](-ang)
    M = np.matmul(np.matmul(D, np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]])), D.T)
    
    if len(test_cov) != 0:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k], field2f
    else:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k]



