import numpy as np
import os 

def prepare_data(directory, pars, krig = False, ellips = True):   
    
    out_dir = os.path.join(directory, 'output')
    
    if ellips:
        files = os.listdir(out_dir)
        matching_files = [file for file in files if file.startswith('covariance_model')]
    
        data = []
        for i in range(len(matching_files)):
            data.append(np.loadtxt(os.path.join(out_dir, matching_files[i])))
    if krig:
        # ppk = np.loadtxt(os.path.join(out_dir, 'pilot_point_k.dat')) 
        pp_cid = np.loadtxt(os.path.join(out_dir, 'pp_cid.dat'))
        k_ref  = np.loadtxt(pars['k_r_d'].replace('Virtual_Reality/',''), delimiter = ',')
        ppk = k_ref[pp_cid.astype(int)] 
        pp_xy = np.loadtxt(os.path.join(out_dir, 'pp_xy.dat'))
    
        ellipsis_data = np.stack(data, axis = 1)
        mean_data = np.mean(ellipsis_data, axis = 1)
        # this should be fixed in later versions of the model
        k_mean = mean_data
    else:
        k_mean = np.genfromtxt(os.path.join(out_dir, 'meanlogk.dat'), delimiter=' ')
    
    errors = np.loadtxt(os.path.join(out_dir, 'errors.dat')) 
    true_obs = np.loadtxt(os.path.join(out_dir, 'obs_true.dat')) 
    mean_obs = np.loadtxt(os.path.join(out_dir, 'obs_mean.dat')) 
    
    
    k_dir = pars['k_r_d'].replace('Virtual_Reality/', '')
    k_true = np.loadtxt(k_dir, delimiter = ',')
    
    if ellips:
        if krig:
            return ellipsis_data, mean_data, errors, ppk, pp_xy, k_mean, k_true, true_obs, mean_obs
        else:
            
            return ellipsis_data, mean_data, errors, [], [], k_mean, k_true, true_obs, mean_obs
    else:
        if krig:
            return [], [], errors, ppk, pp_xy, k_mean, k_true, true_obs, mean_obs
        else:
            
            return [], [], errors, [], [], k_mean, k_true, true_obs, mean_obs