import numpy as np
from dependencies.conditional_k import conditional_k
from dependencies.Kriging import Kriging
from dependencies.k_initial import K_initial

def create_k_fields(gwf, pars: dict, k_ref, pp_xy=[], pp_cid=[], test_cov=[], conditional=True, random=True):
    
    #%% Loading parameters
    clx = pars['lx']          # reference correlation length
    angles = pars['ang']      # reference anisotropy angle
    sigma = pars['sigma'][0]  # reference variance
    mu = pars['mu'][0]        # reference k mean
    dx = pars['dx']           # model discretization
    mg = gwf.modelgrid        # modelgrid
    xyz = mg.xyzcellcenters   # xyz coordinates of cells
    cxy = np.vstack((xyz[0], xyz[1])).T # reshaping (x,y) coordinates
    sig_meas = pars['sig_me'] # measurement uncertainty
    mean_range = np.log(3)    # range from which to draw the mean

    #%% Determining correlation initialization type
    # good initial
    if pars['covtyp'] == "good":
        lx = clx[0] # correlation lengths identical to reference 
        ang = np.random.uniform(0, np.pi) # anisotropy angle is random
        if not random:
            ang = np.deg2rad(angles[0]) # if not random - use reference
    # random initial - factor defines the upper bound of the distribution
    elif 'random' in pars['covtyp']:
        if pars['covtyp'] == 'random_low':
            factor = 0.33 
        elif pars['covtyp'] == 'random_high':
            factor = 0.50
            
        # initial correlation length drawn from [5*cell_size, factor*minor domain length]
        lx = np.array([np.random.randint(pars['dx'][0]*5, np.min(pars['nx'] * pars['dx'])*factor),
                       np.random.randint(pars['dx'][1]*5, np.min(pars['nx'] * pars['dx'])*factor)])
        ang = np.random.uniform(0, np.pi)
    
    # ensure correlation lengths are not identical 
    if lx[0] == lx[1]:
        lx[0] += 1 
               
    if test_cov: # for testing
        lx = test_cov[0]
        ang = test_cov[1]
        
    if pars['valtyp'] == 'good':
        pp_k = np.log(np.squeeze(k_ref)[pp_cid]) 
        pp_k = pp_k + sig_meas * np.random.randn(*pp_k.shape)
    else:
        if pars['valtyp'] == 'random':
            mu = mu
        elif pars['valtyp'] == 'random_low':
            mu = mu - np.log(5) # decrease mean by factor of 5
        elif pars['valtyp'] == 'random_high':
            mu = mu + np.log(5) # increase mean by factor of 5
        
        low_bound, high_bound = mu + np.array([-mean_range,mean_range])
        # std = np.sqrt(pars['sigma'][0])
        # pp_k = np.random.normal(mu, std, len(pp_cid))
        
    mean_val = np.random.uniform(low_bound, high_bound)
    
    
    if pars['f_meas']:
        true_ppk = np.log(np.squeeze(k_ref)[pp_cid.astype(int)]) 
        true_ppk_pert = true_ppk + 0.05 * np.random.randn(*pp_k.shape) * true_ppk
        # CONTINUE HERE
        pp_loc_meas = pars['f_m_id']
        pp_k[] = true_ppk_pert[pars['f_m_id']]
        
        field, field2f = conditional_k(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy)
    else:
        K, K_pp = K_initial(lx, ang, mean_val, sigma, pars, pp_loc = pp_xy)

    
    benchmark_field, _ = Kriging(cxy, dx, lx, pars['ang'][0], pars['sigma'][0], pars, np.log(np.squeeze(k_ref)[pp_cid.astype(int)]), pp_xy)
    
    D = pars['rotmat'](ang)
    M = np.matmul(np.matmul(D, np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]])), D.T)
    
    if len(test_cov) != 0:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k], field2f
    else:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k], benchmark_field

