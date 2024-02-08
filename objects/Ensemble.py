from joblib import Parallel, delayed
import numpy as np
import warnings

# Suppress DeprecationWarning temporarily
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
class Ensemble:
    
    def __init__(self, members: list, nprocs: int, pp_cid, pp_xy, obs_cid: list):
        self.members    = members
        self.nprocs     = nprocs
        self.n_mem      = len(self.members)
        self.pp_cid     = pp_cid
        self.pp_xy      = pp_xy
        self.obs_cid    = obs_cid
        
    def set_field(self, field, pkg_name: list):
        Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].set_field)(
            field[idx],
            pkg_name) 
            for idx in range(self.n_mem)
            )
        
    def propagate(self):
        Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].simulation)(
            ) 
            for idx in range(self.n_mem)
            )
        
    def update_initial_heads(self):
        Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].update_ic)(
            ) 
            for idx in range(self.n_mem)
            )
        
    def get_Kalman_X(self, params: list):   
        head =  Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].get_field)(
            ['h','chd']
            ) 
            for idx in range(self.n_mem)
            )
        
        data = Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].get_field)(
            params
            ) 
            for idx in range(self.n_mem)
            )
        
        # account for fixed head cells --> need to be ommited
        for i in range(self.n_mem):
            chd_cid = list(map(lambda tup: tup[1], head[i]['chd'][0]['cellid']))
            head[i]['h'] = np.delete(head[i]['h'], chd_cid)
        
        # number of states
        nx  = head[0]['h'].size
        for name in params:
            if name == 'npf':
                nx += self.pp_cid.size
            if name == 'cov_data':
                nx += np.array(data[0]['cov_data']).size
        
        X = np.zeros((nx,self.n_mem))
        
        # obtaining k_values at pilot points
        for i in range(self.n_mem):
            if 'cov_data' in params:
                if 'npf' in params:
                    x = np.concatenate((data[i]['cov_data'].flatten(),
                                        data[i]['npf'][:,self.pp_cid].flatten(),
                                        head[i]['h'].flatten()))
                else:
                    x = np.concatenate((data[i]['cov_data'].flatten(),
                                        head[i]['h'].flatten()))
            else:
                x = np.concatenate((data[i]['npf'][:,self.pp_cid].flatten(),
                                    head[i]['h'].flatten()))
                    
            X[:,i] = x
        
        
        Ysim = True
        return X, Ysim
        
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        