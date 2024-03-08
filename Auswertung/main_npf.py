import os
import sys
import importlib.util
sys.path.append('..')
from dependencies.load_template_model import load_template_model
import numpy as np
from dependencies.plot import  plot_k_fields
# from dependencies.prepare_data import prepare_ellipsis_data
# from Clean.dependencies.model_params import get

if __name__ == '__main__':
    
    # pars = get()
    cwd = os.getcwd()
    # specify which folder to investigate
    # target_folder = '/n280_cov_npf_binnac'
    # models = ['np560l1d01',
    #            'np560l01d01',
    #            'np560l02d025']
    models = ['np560l01d01']
    
    # folders = ['dependencies', 'model_data', 'output', 'template_model']
    for target_folder in models:
        model_directory = os.path.join(cwd, target_folder, 'output')
        
        # Import module dynamically based on folder name
        module_name = f"{target_folder}.dependencies.model_params"
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(target_folder, "dependencies", "model_params.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_dir = os.path.join(target_folder, 'template_model')
        
        pars = module.get()
        sim, gwf = load_template_model(pars, model_dir)
        
        k_mean = np.genfromtxt(os.path.join(target_folder, 'output', 'k_mean.dat'), delimiter=' ')
        k_dir = pars['k_r_d'].replace('Virtual_Reality/', '')
        k_true = np.loadtxt(k_dir, delimiter = ',')
        
        for i in range(50):
            plot_k_fields(gwf, [k_mean[i*2], k_true])
        # ellipsis_data, mean_ellipsis, errors = prepare_ellipsis_data(target_directory)
        
    
        # ellipsis(ellipsis_data, mean_ellipsis, errors, pars, target_directory, movie = True)