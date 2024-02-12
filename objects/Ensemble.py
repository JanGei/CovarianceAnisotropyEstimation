from joblib import Parallel, delayed
import numpy as np
import warnings

# Suppress DeprecationWarning temporarily
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    
class Ensemble:
    
    def __init__(self, members: list, nprocs: int, pp_cid, pp_xy, obs_cid: list, mask):
        self.members    = members
        self.nprocs     = nprocs
        self.n_mem      = len(self.members)
        self.pp_cid     = pp_cid
        self.pp_xy      = pp_xy
        self.obs_cid    = [int(i) for i in obs_cid]
        self.ny         = len(obs_cid)
        self.h_mask     = mask.astype(bool)
        
        
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
    
    def apply_X(self, params: list, X):
        # THIS STILL NEEDS A GOOD DEAL OF WORK
        # get head file, identify chd cells and put new 
        head =  Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].get_field)(
            ['h']
            ) 
            for idx in range(self.n_mem)
            )
        
        cov_data = []
        pp_k = []
        heads = []
        
        # THIS WORKS NOW, CONTUNUE HERE
        for i in range(self.n_mem):
            if 'cov_data' in params:
                if 'npf' in params:
                    cov_data.append(X[0:4,i])
                    pp_k.append(X[4:len(self.pp_cid),i])
                    
                    h_interim = head[i]['h'].copy().flatten()
                    h_interim[~self.h_mask] =  X[4+len(self.pp_cid):,i]
                    head[i]['h'] =  h_interim
       
                    # TH
                    # head =
                    # heads.append(np.insert(head, chd_cid, chd_val))
                    # x = np.concatenate((data[i]['cov_data'].flatten(),
                    #                     data[i]['npf'][:,self.pp_cid].flatten(),
                    #                     head[i]['h'].flatten()))
            #     else:
            #         x = np.concatenate((data[i]['cov_data'].flatten(),
            #                             head[i]['h'].flatten()))
            # else:
            #     x = np.concatenate((data[i]['npf'][:,self.pp_cid].flatten(),
            #                         head[i]['h'].flatten()))
        
        
        pass
    
    def get_Kalman_X_Y(self, params: list):   
        head =  Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].get_field)(
            ['h']
            ) 
            for idx in range(self.n_mem)
            )
        
        data = Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].get_field)(
            params
            ) 
            for idx in range(self.n_mem)
            )
        
        Ysim = np.zeros((self.ny,self.n_mem))
        # account for fixed head cells --> need to be ommited
        
        for i in range(self.n_mem):
            Ysim[:,i] = head[i]['h'].flatten()[self.obs_cid]
            head[i]['h'] = head[i]['h'].flatten()[~self.h_mask]
        
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
                                        head[i]['h']))
                else:
                    x = np.concatenate((data[i]['cov_data'].flatten(),
                                        head[i]['h']))
            else:
                x = np.concatenate((data[i]['npf'][:,self.pp_cid].flatten(),
                                    head[i]['h']))
                    
            X[:,i] = x

        return X, Ysim
    
    def update_transient_data(self, rch_data, wel_data, riv_data):
        
        spds = self.members[0].get_field(['rch', 'wel', 'riv'])
        
        rch_spd = spds['rch']
        wel_spd = spds['wel']
        riv_spd = spds['riv']
        
        rivhl = np.ones(np.shape(riv_spd[0]['cellid']))
        
        rch_spd[0]['recharge'] = rch_data
        riv_spd[0]['stage'] = rivhl * riv_data
        wel_spd[0]['q'] = wel_data
        
        Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].set_field)(
            [rch_spd, wel_spd, riv_spd],
            ['rch', 'wel', 'riv']
            ) 
            for idx in range(self.n_mem)
            )
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        