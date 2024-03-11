import os
import sys
sys.path.append('..')
from dependencies.plot import ellipsis, ellipsis_test
import importlib.util
from dependencies.prepare_data import prepare_ellipsis_data
from dependencies.krig_field import krig
# from Clean.dependencies.model_params import get
import flopy

if __name__ == '__main__':
    
    # pars = get()
    cwd = os.getcwd()
    # specify which folder to investigate
    folder = '3types'
    models = ['xl02d015d075']
    
    for model in models:
        target_folder = os.path.join(cwd, folder, model)
        model_directory = os.path.join(target_folder, 'output')
        
        krig = False
        test = True
        
        dep_dir = os.path.join(cwd, target_folder, 'dependencies')
        module_name1 = f"{dep_dir}.model_params"
        spec1 = importlib.util.spec_from_file_location(module_name1, os.path.join(dep_dir, "model_params.py"))
        module1 = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(module1)
        pars = module1.get()
        
        if krig:
            # NEEDS MORE DEVELOPMENT
            module_name2 = f"{dep_dir}.load_template_model"
            spec2 = importlib.util.spec_from_file_location(module_name2, os.path.join(dep_dir, "load_template_model.py"))
            module2 = importlib.util.module_from_spec(spec2)
            spec2.loader.exec_module(module2)
            _ , gwf = module2.load_template_model(pars)
            
            
            ellipsis_data, mean_ellipsis, errors, ppk = prepare_ellipsis_data(target_folder, krig = krig)
            ellipsis(ellipsis_data, mean_ellipsis, errors, pars, model_directory, movie = True)
        elif test:
            ellipsis_data, mean_ellipsis, errors, _ = prepare_ellipsis_data(target_folder, krig = krig)
            ellipsis_test(ellipsis_data, mean_ellipsis, errors, pars)
        else:
            ellipsis(ellipsis_data, mean_ellipsis, errors, pars, model_directory, movie = True)
    


# (cov_data, mean_cov, pars)