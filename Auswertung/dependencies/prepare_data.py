import numpy as np
import os 

def prepare_ellipsis_data(directory, krig = False):   
    
    out_dir = os.path.join(directory, 'output')
    
    files = os.listdir(out_dir)
    matching_files = [file for file in files if file.startswith('covariance_model')]
    
    data = []
    for i in range(len(matching_files)):
        data.append(np.loadtxt(os.path.join(out_dir, matching_files[i])))
    
    
    ellipsis_data = np.stack(data, axis = 1)
    mean_data = np.mean(ellipsis_data, axis = 1)
    
    errors = np.loadtxt(os.path.join(out_dir, 'errors.dat')) 
    
    
    
    if krig == False:
        return ellipsis_data, mean_data, errors, []
    else:
        ppk = np.loadtxt(os.path.join(out_dir, 'pilot_point_k.dat')) 
        return ellipsis_data, mean_data, errors, ppk
