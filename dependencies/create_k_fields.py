import gstools as gs
from gstools import krige
import numpy as np

def create_k_fields(gwf, pars: dict, pp_xy, pp_cid: np.ndarray, covtype = 'random', valtype = 'good'):
    dim = 2
    cov = pars['cov']
    lx      = pars['lx']
    ang     = pars['ang']
    sigma   = pars['sigma'][0]
    # mu      = pars['mu']
    cov     = pars['cov']
    k_ref   = pars['k_ref']
    n_mem   = pars['n_mem']

    mg = gwf.modelgrid
    xyz = mg.xyzcellcenters
    
    k_fields = []
    for i in range(n_mem):
        if covtype == 'random':
            lx = np.array([np.random.randint(100, 2000), np.random.randint(100, 2000)])
            ang = np.random.randint(0, 360)
        elif covtype == 'good':
            lx = lx[0]
            ang = ang[0]
            pass
        
        
        if cov == 'Matern':
            model = gs.Matern(dim=dim, var=sigma, angles = ang, len_scale=lx)
        elif cov == 'Exponential':
            model = gs.Exponential(dim = dim, var = sigma, len_scale=lx, angles = ang)
        elif cov == 'Gaussian':
            model = gs.Gaussian(dim = dim, var = sigma, len_scale=lx, angles = ang)
        
        # starting k values at pilot points
        if valtype == 'good':
            pp_k = k_ref[pp_cid.astype(int)]
        
        krig = krige.Ordinary(model, cond_pos=(pp_xy[:,0], pp_xy[:,1]), cond_val = np.log(pp_k))
        field = krig((xyz[0], xyz[1]))
        k_fields.append(np.exp(field[0]))
    return k_fields
    
        


