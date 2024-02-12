import numpy as np
def load_observations(pars: dict):
    
    observations = np.load(pars['vr_o_d'], allow_pickle=True).tolist()
    
    return observations