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
        factor = 1
    elif pars['estyp'] == "good":
        factor = 0.5
    elif pars['estyp'] == "underestimate":
        factor = 0.25
    
    if pars['covt'] == 'random':
        lx = np.array([np.random.randint(pars['dx'][0]*3, np.min(pars['nx'] * pars['dx'])*factor),
                       np.random.randint(pars['dx'][1]*3, np.min(pars['nx'] * pars['dx'])*factor/3)])
        ang = np.random.uniform(-np.pi/2, np.pi/2)
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
        std = np.sqrt(pars['sigma'][0])
        pp_k = np.random.normal(mu, std, len(pp_cid)) 
    
    #correcting a few pilot points if measurements are available
    if pars['f_meas']:
        true_ppk = np.log(k_ref[pp_cid.astype(int)]) 
        true_ppk_pert = true_ppk + 0.05 * np.random.randn(*pp_k.shape) * true_ppk
        pp_k[pars['f_m_id']] = true_ppk_pert[pars['f_m_id']]

    
    if conditional:
        field, field2f = conditional_k(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy)
    else:
        field, field2f = Kriging(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy)
    
    # The ellips is rotated counter-clockwise
    D = pars['rotmat'](ang)
    M = np.matmul(np.matmul(D, np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]])), D.T)
    
    if len(test_cov) != 0:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k], field2f
    else:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k]



