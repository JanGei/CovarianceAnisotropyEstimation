import os
import sys
import importlib.util
sys.path.append('..')
from dependencies.load_template_model import load_template_model
from dependencies.plotFH import  plot_k_fields
from dependencies.initial_fields import plot_initial_k_fields
from dependencies.ellips_k import  ellips_k
from dependencies.prepare_data import prepare_data
import numpy as np
# from dependencies.plot import  plot_k_fields
# from dependencies.prepare_data import prepare_ellipsis_data
# from Clean.dependencies.model_params import get

if __name__ == '__main__':
    cwd = os.getcwd()
    # specify which folder to investigate
    folder = 'icnpp'
    models = ['IC560l02d025']
    # models = ['nPP560l1d05']
    
    for model in models:
        target_folder = os.path.join(cwd, folder, model)
        model_directory = os.path.join(target_folder, 'output')
        
        # Import module dynamically based on folder name
        module_name = f"{target_folder}.dependencies.model_params"
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(target_folder, "dependencies", "model_params.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_dir = os.path.join(target_folder, 'template_model')
        
        pars = module.get()
        sim, gwf = load_template_model(pars, model_dir)
        

        if pars['pilotp']:
            if 'cov_data' in pars['EnKF_p']:
                ellipsis_data, mean_data, errors, ppk, pp_xy, k_mean, k_true, true_obs, mean_obs, k_ini = prepare_data(target_folder, pars, krig = True, ellips = True)
            else:
                _, _, errors, ppk, pp_xy, k_mean, k_true, true_obs, mean_obs, k_ini = prepare_data(target_folder, pars, krig = True, ellips = False)
        else:
            if 'cov_data' in pars['EnKF_p']:
                ellipsis_data, mean_data, errors, _, _, k_mean, k_true, true_obs, mean_obs, k_ini = prepare_data(target_folder, pars, krig = False, ellips = True)
            else:
                _, _, errors, _, _,k_mean, k_true, true_obs, mean_obs, k_ini = prepare_data(target_folder, pars, krig = False, ellips = False)
                
        plot_initial_k_fields(gwf, pars, k_ini)
        # plot_k_fields(gwf, pars, [k_mean, k_true], true_obs, mean_obs, model_directory, movie = True)
        # ellipsis_data, mean_ellipsis, errors = prepare_ellipsis_data(target_directory)
        # ellips_k(gwf, pars, ellipsis_data, mean_data, k_true, ppk, pp_xy, model_directory, movie=True)
        
    
        # ellipsis(ellipsis_data, mean_ellipsis, errors, pars, target_directory, movie = True)