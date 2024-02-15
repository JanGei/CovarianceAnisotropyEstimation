import numpy as np
import os 
def prepare_ellipsis_data(directory):
    
    files = os.listdir(directory)
    matching_files = [file for file in files if file.startswith('covariance_model')]
    
    data = []
    for i in range(len(matching_files)):
        data.append(np.loadtxt(directory + '/' + matching_files[i]))
    
    ellipsis_data = np.stack(data, axis = 1)
    mean_data = np.mean(ellipsis_data, axis = 1)
    
    return ellipsis_data, mean_data