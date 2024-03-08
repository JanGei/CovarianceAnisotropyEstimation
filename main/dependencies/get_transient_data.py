import numpy as np

def get_transient_data(pars: dict, t_step: int, true_h, obs_cid):
    

    r_ref = np.loadtxt(pars['r_r_d'], delimiter = ',')
    sfac  = np.genfromtxt(pars['sf_d'],delimiter = ',', names=True)['Wert']
    welst = pars['welst'] 
    welnd = pars['welnd'] 
    welq  = pars['welq'] 
    rivh  = np.genfromtxt(pars['rh_d'],delimiter = ',', names=True)['Wert']

    rch_data = abs(np.array(r_ref).flatten()) * sfac[t_step]
    wel_data = np.zeros(5)
    time = 0.25 * t_step
    for i in range(len(welq)):
        if welst[i] <= time and welnd[i] > time:
            wel_data[i] = -welq[i]
    riv_data = rivh[t_step]
    
    # Y_obs = np.ones(len(obs_cid))
    for key in  obs_cid:
        Y_obs = np.array(true_h).flatten()[obs_cid.tolist()]
        
    return rch_data, wel_data, riv_data, Y_obs