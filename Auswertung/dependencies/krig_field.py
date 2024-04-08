import gstools as gs
from gstools import krige

def krig(pars, gwf, ang, lx):
    
    xyz = gwf.modelgrid
    
    if pars['cov'] == 'Matern':
        model = gs.Matern(dim=2, var=pars['sigma'][0], angles = ang, len_scale=lx)
    elif pars['cov'] == 'Exponential':
        model = gs.Exponential(dim = 2, var = pars['sigma'][0], len_scale=lx, angles = ang)
    elif pars['cov'] == 'Gaussian':
        model = gs.Gaussian(dim = 2, var = pars['sigma'][0], len_scale=lx, angles = ang)
        
    krig = krige.Ordinary(model, cond_pos=(pp_xy[:,0], pp_xy[:,1]), cond_val = np.log(pp_k))
    field = krig((xyz[0], xyz[1]))
    
    return field 