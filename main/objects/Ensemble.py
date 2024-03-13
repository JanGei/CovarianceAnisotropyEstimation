from joblib import Parallel, delayed
import numpy as np
import os
import time
    
class Ensemble:
    
    def __init__(self, members: list, pilotp_flag, nprocs: int, obs_cid: list, mask, ellipses = [],pp_cid = [], pp_xy = []):
        self.members    = members
        self.nprocs     = nprocs
        self.n_mem      = len(self.members)
        self.obs_cid    = [int(i) for i in obs_cid]
        self.ny         = len(obs_cid)
        self.h_mask     = mask.astype(bool)
        self.ole        = []
        self.ole_nsq    = []
        self.te1        = []
        self.te1_nsq    = []
        self.te2        = []
        self.te2_nsq    = []
        self.obs        = []
        self.pilotp_flag= pilotp_flag
        self.meanlogk   = []
        self.logmeank   = []
        if pilotp_flag:
            self.ellipses   = ellipses
            self.pp_cid     = pp_cid
            self.pp_xy      = pp_xy
            self.mean_cov   = np.mean(ellipses, axis = 0)
            self.meanlogppk = []
            self.logmeanppk = []
        
        
    def set_field(self, field, pkg_name: list):
        Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].set_field)(
            [field[idx]],
            pkg_name) 
            for idx in range(self.n_mem)
            )
        
    def propagate(self):
        Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].simulation)(
            ) 
            for idx in range(self.n_mem)
            )
        
    def update_initial_heads(self):
        Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].update_ic)(
            ) 
            for idx in range(self.n_mem)
            )
        
    def get_damp(self, X, val, params):
        damp = np.zeros((X[:,0].size)) + val[0]
        if 'cov_data' in params:
            cl = len(np.unique(self.members[0].ellips_mat))
            damp[:cl] = val[1]
            if 'npf' in params:
                damp[cl:cl+len(self.pp_cid)] = val[2]
        else:
            if self.pilotp_flag:
                damp[:len(self.pp_cid)] = val[1]
            else:
                damp[:len(self.members[0].npf.k.array.squeeze())] = val[1]
            
        return damp
        
    def write_simulations(self):
        Parallel(n_jobs=self.nprocs,
                 backend="threading")(
                     delayed(self.members[idx].write_sim)()
                     for idx in range(self.n_mem)
                     )
        
    def apply_X(self, params: list, X):

        head = self.get_member_fields(['h'])
        
        data = []
        
        # Sort the corrected data
        for i in range(self.n_mem):
            if 'cov_data' in params:
                cl = 3
                if 'npf' in params:
                    head[i]['h'] = head[i]['h'].flatten()
                    head[i]['h'][~self.h_mask] = X[cl+len(self.pp_cid):,i]
                    
                    data.append([X[:cl,i], X[cl:len(self.pp_cid)+cl,i]])
                      
                else:
                    head[i]['h'] = head[i]['h'].flatten()
                    head[i]['h'][~self.h_mask] = X[cl:,i]

                    data.append([X[:cl,i], np.log(self.members[0].npf.k.array.flatten()[self.pp_cid])])
                    
            else:
                head[i]['h'] = head[i]['h'].flatten()
                
                
                if self.pilotp_flag:
                    head[i]['h'][~self.h_mask] = X[len(self.pp_cid):,i]
                    data.append(X[:len(self.pp_cid),i])
                else:
                    head[i]['h'][~self.h_mask] = X[self.members[0].npf.k.array.size:,i]
                    data.append(X[:self.members[0].npf.k.array.size,i])
            
            
        
        if self.pilotp_flag:
            start_time = time.time()
            result = Parallel(n_jobs=self.nprocs,
                              backend="threading")(
                                  delayed(self.members[idx].kriging)(
                                      params, 
                                      data[idx], 
                                      self.pp_xy,
                                      self.pp_cid
                                      ) 
                                  for idx in range(self.n_mem)
                                  )
            print(f'Kriging alone took {(time.time() - start_time):.2f} seconds')
            Parallel(n_jobs=self.nprocs,
                     backend="threading")(
                         delayed(self.members[idx].set_field)(
                             [head[idx]['h']], ['h']
                             ) 
                         for idx in range(self.n_mem)
                         )
        else:
            Parallel(n_jobs=self.nprocs,
                     backend="threading")(
                         delayed(self.members[idx].set_field)(
                             [head[idx]['h'],np.exp(data[idx])], ['h', 'npf']
                             ) 
                         for idx in range(self.n_mem)
                         )

        if 'cov_data' in params:
            self.ellipses = np.array(result)
            self.mean_cov = self.sort_ellipses(result)
            if 'npf' in params:
                self.meanlogppk = np.mean(X[cl:len(self.pp_cid)+cl,:], axis = 1)
                self.logmeanppk = np.log(np.mean(np.exp(X[cl:len(self.pp_cid)+cl,:]), axis = 1))
        else:
            if self.pilotp_flag:
                self.meanlogppk = np.mean(X[:len(self.pp_cid),:], axis = 1)
                self.logmeanppk = np.log(np.mean(np.exp(X[:len(self.pp_cid),:]), axis = 1))
                

    
    def sort_ellipses(self, data):
        placeholder = np.ones(3)
        data = np.array(data)
        sorted_data = []
        # Step 1 - allign on major axis
        for i in range(data.shape[0]):
            if data[i,0] < data[i,1]:
                placeholder[0] = data[i,1]
                placeholder[1] = data[i,0]
                placeholder[2] = data[i,2] + np.pi/2
            else:
                placeholder[0] = data[i,0]
                placeholder[1] = data[i,1]
                placeholder[2] = data[i,2] 
                
        # Step 2 - allign on upper two quadrants
            placeholder[2] = placeholder[2]%np.pi

            sorted_data.append(placeholder.copy())        
            
        
        return np.mean(np.array(sorted_data), axis = 0)
    

                    
    def get_Kalman_X_Y(self, params: list):   

        head = self.get_member_fields(['h'])
        data = self.get_member_fields(params)
        
        
        Ysim = np.zeros((self.ny,self.n_mem))
        # account for fixed head cells --> need to be ommited
        
        for i in range(self.n_mem):
            Ysim[:,i] = head[i]['h'].flatten()[self.obs_cid]
            head[i]['h'] = head[i]['h'].flatten()[~self.h_mask]
        
        # number of states
        nx  = head[0]['h'].size
        if self.pilotp_flag:
            for name in params:
                if name == 'npf':
                    nx += self.pp_cid.size
                if name == 'cov_data':
                    nx += np.array(data[0]['cov_data']).size
        else:
            nx += self.members[0].npf.k.array.size
        
        X = np.zeros((nx,self.n_mem))
        
        
        # obtaining k_values at pilot points
        for i in range(self.n_mem):
            if 'cov_data' in params:
                if 'npf' in params:
                    # log transformation of cov l to prevent negative numbers?? 
                    x = np.concatenate((data[i]['cov_data'].flatten(),
                                        np.log(data[i]['npf'][:,self.pp_cid].flatten()),
                                        head[i]['h']))
                else:
                    x = np.concatenate((data[i]['cov_data'].flatten(),
                                        head[i]['h']))
            else:
                if self.pilotp_flag:
                    x = np.concatenate((np.log(data[i]['npf'][:,self.pp_cid].flatten()),
                                        head[i]['h']))
                else:
                    x = np.concatenate((np.log(data[i]['npf'].flatten()),
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
        mean_obs = mean_h[self.obs_cid]
        true_obs = true_h[self.obs_cid]
        self.obs = [true_obs, mean_obs]
        
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
        
        self.te1_nsq.append(np.sum(np.square(true_h - mean_h)/var_h))
        self.te2_nsq.append(np.sum(np.square(true_h - mean_h)/var_te2))
        
        te1 = 0
        te2 = 0
        for i in range(len(self.te1_nsq)):
            te1 += self.te1_nsq[i]
            te2 += self.te2_nsq[i]
        
        # nrmse for the model up until the current time step
        self.te1.append(np.sqrt(te1/len(self.te1_nsq)/mean_h.size))
        self.te2.append(np.sqrt(te2/len(self.te2_nsq)/mean_h.size))
    
    def get_member_fields(self, params):
        
        data = Parallel(n_jobs=self.nprocs, backend="threading")(delayed(self.members[idx].get_field)(
            params
            ) 
            for idx in range(self.n_mem)
            )
        
        return data
        

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
    
    def record_state(self, pars: dict, params: list):
        
        k_fields = self.get_member_fields(['npf'])
        k_fields = np.array([field['npf'] for field in k_fields]).squeeze()
        self.meanlogk = np.mean(np.log(k_fields), axis = 0)
        self.logmeank = np.log(np.mean(k_fields, axis = 0))
        
        direc = pars['resdir']
        
        f = open(os.path.join(direc,  'errors.dat'),'a')
        f.write("{:.4f} ".format(self.ole[-1]))
        f.write("{:.4f} ".format(self.te1[-1]))
        f.write("{:.4f} ".format(self.te2[-1]))
        f.write('\n')
        f.close()
        
        f = open(os.path.join(direc,  'obs_true.dat'),'a')
        g = open(os.path.join(direc,  'obs_mean.dat'),'a')
        for i in range(len(self.obs[0])):
            f.write("{:.5f} ".format(self.obs[0][i]))
            g.write("{:.5f} ".format(self.obs[1][i]))
        f.write('\n')
        g.write('\n')
        f.close()
        g.close()
        
        
        
        # also store covariance data for all models
        if 'cov_data' in params:
            cov_data = self.get_member_fields(['cov_data'])
            
            f = open(os.path.join(direc, 'covariance_data.dat'),'a')
            f.write("{:.10f} ".format(self.mean_cov[0]))
            f.write("{:.10f} ".format(self.mean_cov[1]))
            f.write("{:.10f} ".format(self.mean_cov[2]))
            f.write('\n')
            f.close()
            
            for i in range(self.n_mem):
                f = open(os.path.join(direc, f'covariance_model_{i}.dat'), 'a')
                for j in range(len(cov_data[i]['cov_data'])):
                    f.write("{:.10f} ".format(cov_data[i]['cov_data'][j]))
                f.write('\n')
                f.close()

        if 'npf' in params:
            if self.pilotp_flag:
                f = open(os.path.join(direc,  'meanlogppk.dat'),'a')
                for i in range(len(self.meanlogppk)):
                    f.write("{:.8f} ".format(self.meanlogppk[i]))
                f.write('\n')
                f.close()
                
                f = open(os.path.join(direc,  'logmeanppk.dat'),'a')
                for i in range(len(self.logmeanppk)):
                    f.write("{:.8f} ".format(self.logmeanppk[i]))
                f.write('\n')
                f.close()
            
            f = open(os.path.join(direc,  'meanlogk.dat'),'a')
            for i in range(len(self.meanlogk)):
                f.write("{:.8f} ".format(self.meanlogk[i]))
            f.write('\n')
            f.close()
            
            f = open(os.path.join(direc,  'logmeank.dat'),'a')
            for i in range(len(self.logmeank)):
                f.write("{:.8f} ".format(self.logmeank[i]))
            f.write('\n')
            f.close()
        
    def remove_current_files(self, pars):
        
        file_paths = [os.path.join(pars['resdir'], 'errors.dat'),
                      os.path.join(pars['resdir'], 'covariance_data.dat'),
                      os.path.join(pars['resdir'], 'logmeanppk.dat'),
                      os.path.join(pars['resdir'], 'meanlogppk.dat'),
                      os.path.join(pars['resdir'], 'obs_true.dat'),
                      os.path.join(pars['resdir'], 'logmeank.dat'),
                      os.path.join(pars['resdir'], 'meanlogk.dat'),
                      os.path.join(pars['resdir'], 'k_mean.dat'),
                      os.path.join(pars['resdir'], 'obs_mean.dat')]
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        for filename in os.listdir(pars['resdir']):
            # Check if the surname is in the filename
            if 'covariance_model_' in filename:
                # Construct the full file path
                file_path = os.path.join(pars['resdir'], filename)
                # Remove the file
                os.remove(file_path)

        


        
        
        
        
        
        
        
        
        
        
        
        