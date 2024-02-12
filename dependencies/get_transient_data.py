import numpy as np

def get_transient_data(pars: dict, t_step: int, obs_val):
    
    sfac = pars['sfac']
    r_ref = pars['r_ref']
    welst = pars['welst'] 
    welnd = pars['welnd'] 
    welq  = pars['welq'] 
    rivh  = pars['rivh']

    rch_data = abs(np.array(r_ref).flatten()) * sfac[t_step]
    wel_data = np.zeros(5)
    time = 0.25 * t_step
    for i in range(len(welq)):
        if welst[i] <= time and welnd[i] > time:
            wel_data[i] = -welq[i]
    riv_data = rivh[t_step]
    
    Y_obs = np.ones(len(obs_val))
    for key in  obs_val.keys():
        Y_obs[key] = obs_val[key]['h_obs'][0,t_step]
        
    return rch_data, wel_data, riv_data, Y_obs