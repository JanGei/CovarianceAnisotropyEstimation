from joblib import Parallel, delayed
import numpy as np
    
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
        self.ole        = []
        self.ole_nsq    = []
        self.te1        = []
        self.te1_nsq    = []
        self.te2        = []
        self.te2_nsq    = []
        
        
    def set_field(self, field, pkg_name: list):
        Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].set_field)(
            [field[idx]],
            pkg_name) 
            for idx in range(self.n_mem)
            )
        
    def propagate(self):
        Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].simulation)(
            ) 
            for idx in range(self.n_mem)
            )
        
    def update_initial_heads(self):
        Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].update_ic)(
            ) 
            for idx in range(self.n_mem)
            )
    
    def apply_X(self, params: list, X):
        # get head file, identify chd cells and put new 
        head =  Parallel(n_jobs=self.nprocs)(delayed(self.members[idx].get_field)(
            ['h']
            ) 
            for idx in range(self.n_mem)
            )
        
        data = []
        
        # Sort the corrected data
        for i in range(self.n_mem):
            if 'cov_data' in params:
                if 'npf' in params:
                    head[i]['h'] = head[i]['h'].flatten()
                    head[i]['h'][~self.h_mask] = X[4+len(self.pp_cid):,i]
                    
                    data.append([X[0:4,i], X[4:len(self.pp_cid)+4,i]])

                else:
                    head[i]['h'] = head[i]['h'].flatten()
                    head[i]['h'][~self.h_mask] = X[4:,i]

                    data.append([X[0:4,i], self.npf.k.array.flatten()[self.pp_cid]])

            else:
                head[i]['h'] = head[i]['h'].flatten()
                head[i]['h'][~self.h_mask] = X[len(self.pp_cid):,i]
                
                data.append(X[:len(self.pp_cid),i])

        
        Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].set_field)(
            [head[idx]['h']], ['h']
            ) 
            for idx in range(self.n_mem)
            )
        
        Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].kriging)(
            params, data[idx], self.pp_xy
            ) 
            for idx in range(self.n_mem)
            )

     
    
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
                    # log transformation of cov l to prevent negative numbers?? 
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
        
    
    def model_error(self,  true_h):
        
        mean_h, var_h = self.get_mean_var()
        true_h = np.array(true_h).flatten()
        
        # analogous for ole
        mean_obs = mean_h[self.pp_cid]
        true_obs = true_h[self.pp_cid]
        
        self.ole_nsq.append(np.sum(np.square(true_obs - mean_obs)/0.1)/mean_obs.size)
        
        ole = 0
        for i in range(len(self.ole_nsq)):
            ole += self.ole_nsq[i] 
        
        # ole for the model up until the current time step
        self.ole.append(np.sqrt(ole/len(self.ole_nsq)))
        
        # calculating nrmse without root for later summation
        true_h = true_h[~self.h_mask]
        mean_h = mean_h[~self.h_mask]
        var_h = var_h[~self.h_mask]
        var_te2 = (true_h + mean_h)/2
        
        self.te1_nsq.append(np.sum(np.square(true_h - mean_h)/var_h)/mean_h.size)
        self.te2_nsq.append(np.sum(np.square(true_h - mean_h)/var_te2)/mean_h.size)
        
        te1 = 0
        te2 = 0
        for i in range(len(self.te1_nsq)):
            te1 += self.te1_nsq[i]
            te2 += self.te2_nsq[i]
        
        # nrmse for the model up until the current time step
        self.te1.append(np.sqrt(te1/len(self.te1_nsq)))
        self.te2.append(np.sqrt(te1/len(self.te2_nsq)))
        

    def get_mean_var(self):
        h_fields = []
        for member in self.members:
            h_fields.append(member.get_field(['h'])['h'].flatten())
        
        mean_h = np.zeros_like(h_fields[0])
        var_h = np.zeros_like(h_fields[0])
        count = 0
        
        for field in h_fields:
            mean_h += field
            var_h += np.square(field)
            count += 1
            
        mean_h = mean_h/count
        var_h = (var_h / count) - np.square(mean_h)
        
        return mean_h, var_h
        
        
        
        
        
        
        
        
        
        
        
        