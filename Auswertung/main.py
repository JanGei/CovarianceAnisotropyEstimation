import os
import sys
sys.path.append('..')
from dependencies.plot import ellipsis
from dependencies.prepare_data import prepare_ellipsis_data
from Clean.dependencies.model_params import get

if __name__ == '__main__':
    
    pars = get()
    cwd = os.getcwd()
    # specify which folder to investigate
    # target_folder = '/n280_cov_npf_binnac'
    target_folder = 'Matern140v1e002_binac'
    target_directory = os.path.join(cwd, target_folder, 'output')
    
    ellipsis_data, mean_ellipsis = prepare_ellipsis_data(target_directory)
    


    ellipsis(ellipsis_data, mean_ellipsis, pars, target_directory, movie = True)
    


# (cov_data, mean_cov, pars)