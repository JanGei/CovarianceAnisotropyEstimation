import numpy as np
def load_observations(pars: dict):
    
    observations = np.load(pars['vr_o_d'], allow_pickle=True).tolist()
    
    return observations

def load_true_h_field(pars: dict):
    
    true_h = np.load(pars['vr_h_d'], allow_pickle=True).tolist()
    
    return true_h