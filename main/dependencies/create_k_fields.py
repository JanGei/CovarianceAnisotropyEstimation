import numpy as np

def create_k_fields(gwf, pars: dict, k_ref, pp_xy=[], pp_cid=[], test_cov=[], conditional=True, random=True):
    from dependencies.conditional_k import conditional_k
    from dependencies.Kriging import Kriging

    clx = pars['lx']
    angles = pars['ang']
    sigma = pars['sigma'][0]
    dx = pars['dx']
    
    mg = gwf.modelgrid
    xyz = mg.xyzcellcenters
    cxy = np.vstack((xyz[0], xyz[1])).T
    sig_meas = pars['sig_me']

    if pars['covtyp'] == "good":
        lx = clx[0]
        ang = np.random.uniform(0, np.pi)
        if not random:
            lx = clx[0]
            ang = np.deg2rad(angles[0])
    else:
        if pars['covtyp'] == 'random':
            factor = 0.5
        elif pars['covtyp'] == 'random_low':
            factor = 0.33
        elif pars['covtyp'] == 'random_high':
            factor = 0.66
            
        lx = np.array([np.random.randint(pars['dx'][0]*5, np.min(pars['nx'] * pars['dx'])*factor),
                       np.random.randint(pars['dx'][1]*5, np.min(pars['nx'] * pars['dx'])*factor)])
        ang = np.random.uniform(0, np.pi)
    

    if lx[0] == lx[1]:
        lx[0] += 1
               
    if test_cov:
        lx = test_cov[0]
        ang = test_cov[1]
        
    if pars['valtyp'] == 'good':
        pp_k = np.log(np.squeeze(k_ref)[pp_cid]) 
        pp_k = pp_k + sig_meas * np.random.randn(*pp_k.shape)
    else:
        if pars['valtyp'] == 'random':
            mu = np.mean(np.log(k_ref)) 
        elif pars['valtyp'] == 'random_low':
            mu = np.mean(np.log(k_ref)) / 0.7
        elif pars['valtyp'] == 'random_high':
            mu = np.mean(np.log(k_ref)) * 0.7
            
        std = np.sqrt(pars['sigma'][0])
        pp_k = np.random.normal(mu, std, len(pp_cid))
    
    
    if pars['f_meas']:
        true_ppk = np.log(np.squeeze(k_ref)[pp_cid.astype(int)]) 
        true_ppk_pert = true_ppk + 0.05 * np.random.randn(*pp_k.shape) * true_ppk
        pp_k[pars['f_m_id']] = true_ppk_pert[pars['f_m_id']]

    if conditional:
        field, field2f = conditional_k(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy)
    else:
        field, field2f = Kriging(cxy, dx, lx, ang, sigma, pars, pp_k, pp_xy)
    
    benchmark_field, _ = Kriging(cxy, dx, lx, pars['ang'][0], pars['sigma'][0], pars, np.log(np.squeeze(k_ref)[pp_cid.astype(int)]), pp_xy)
    
    D = pars['rotmat'](ang)
    M = np.matmul(np.matmul(D, np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]])), D.T)
    
    if len(test_cov) != 0:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k], field2f
    else:
        return field, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang], [pp_xy, pp_k], benchmark_field




