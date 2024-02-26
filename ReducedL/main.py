from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from dependencies.convert_transient import convert_to_transient
from dependencies.create_pilot_points import create_pilot_points
from dependencies.create_k_fields import create_k_fields
from dependencies.load_template_model import load_template_model
from dependencies.load_observations import load_true_h_field
from dependencies.get_transient_data import get_transient_data
from dependencies.plot import plot_fields, plot_POI, plot_k_fields, plot, ellipsis
from dependencies.intersect_with_grid import intersect_with_grid
from dependencies.generate_mask import chd_mask
from objects.Ensemble import Ensemble
from objects.MFModel import MFModel
from objects.EnsembleKalmanFilter import EnsembleKalmanFilter
import time
import numpy as np
from joblib import Parallel, delayed

import warnings

# Define a filter function to suppress specific warnings
def joblib_warning_filter(message, category, filename, lineno, file=None, line=None):
    if "joblib" in str(filename):
        return None  # Suppress the warning
    else:
        return message, category, filename, lineno, None, None


if __name__ == '__main__':

    # Register the filter function with the warnings module
    warnings.showwarning = joblib_warning_filter
    
    pars        = get()
    n_mem       = pars['n_mem']
    nprocs      = pars['nprocs']
    
    if pars['up_tem']:
        temp_sim = convert_to_transient(pars['tm_ws'], pars)
    
    print(f'Joblib initiated with {nprocs} processors')
    print(f'The template model is located in {pars["tm_ws"]}')
    #%% loading necessary data
    start_time = time.time()
    
    # copy template model to ene
    model_dir   = create_Ensemble(pars)
    sim, gwf = load_template_model(pars)
    

    pp_cid, pp_xy = create_pilot_points(gwf, pars)
    
    obs_cid = intersect_with_grid(gwf, pars['obsxy'])
    # not needed here, but will be needed in non synthetic cases
    # obs_val = load_observations(pars) 
    true_h =  load_true_h_field(pars)
    
    if 'cov_data' in pars['EnKF_p']:
        covtype = "random"
    else:
        covtype = "good"
    if 'npf' in pars['EnKF_p']:
        valtype = "random"
    else:
        valtype = "good"
    
    result = Parallel(n_jobs=nprocs)(delayed(create_k_fields)(
        gwf,
        pars, pp_xy,
        pp_cid,
        covtype = covtype,
        valtype = valtype) 
        for idx in range(n_mem)
        )
    
    k_fields = []
    cov_models = []
    cor_ellips = []
    # sorting the results
    for tup in result:
        field, covmod, ellips = tup
        k_fields.append(field)
        cov_models.append(covmod)
        cor_ellips.append(ellips)
    
    # plot_POI(gwf, pp_xy, pars, bc = True)
    # plot_fields(gwf, pars,  k_fields[0], k_fields[1])
    plot_k_fields(gwf, pars,  k_fields)
    # plot(gwf, ['logK','h'], bc=True)
    
    mask_chd = chd_mask(gwf)
    
    print(f'The model has {len(obs_cid)} observation points')
    print(f'The model has {len(pp_cid)} pilot points points')
    print(f'Loading of data and creating k_fields took {(time.time() - start_time):.2f} seconds')
    
    #%% generate model instances  
    start_time = time.time()
    models = Parallel(n_jobs=nprocs, backend="threading")(delayed(MFModel)(
        model_dir[idx],
        pars['mname'],
        cov_models[idx],
        cor_ellips[idx]) 
        for idx in range(n_mem)
        )
    
    print(f'{n_mem} models are initiated in {(time.time() - start_time):.2f} seconds')
    #%% add the models to the ensemble
    start_time = time.time()
    
    MF_Ensemble     = Ensemble(models,
                               np.mean(np.array(cor_ellips), axis = 0),
                               nprocs,
                               pp_cid,
                               pp_xy,
                               obs_cid,
                               mask_chd)
    
    # set their respective k-fields
    MF_Ensemble.set_field(k_fields, ['npf'])
    
    print(f'Ensemble is initiated and respective k-fields are set in {(time.time() - start_time):.2f} seconds')
    #%% Running each model 10 times
    start_time = time.time()
    
    for idx in range(pars['nprern']):
        MF_Ensemble.propagate()
        MF_Ensemble.update_initial_heads()
    # print(MF_Ensemble.get_mean_var())
    
    print(f'Each model is run and updated {pars["nprern"]} times which took {(time.time() - start_time):.2f} seconds')
    print(f'That makes {((time.time() - start_time)/(pars["nprern"] * n_mem)):.2f} seconds per model run')
    #%%
    X, Ysim = MF_Ensemble.get_Kalman_X_Y(pars['EnKF_p'])
    damp = MF_Ensemble.get_damp(X, pars['damp'],pars['EnKF_p'])
    EnKF = EnsembleKalmanFilter(X, Ysim, damp = damp, eps = pars['eps'])
    
    covl = []
    k_means = []
    Assimilate = True
    # for t_step in range(pars['nsteps']):
    for t_step in range(24):
        if t_step == 0:
            MF_Ensemble.remove_current_files(pars)
        if t_step == 1200:
            Assimilate = False
        
        # visualize covariance structures
        if pars['setup'] == 'office':
            covl.append(MF_Ensemble.get_member_fields(['cov_data'])[0])
            ellipsis(
                MF_Ensemble.get_member_fields(['cov_data']),
                MF_Ensemble.mean_cov,
                pars
                )
            # if t_step%5 == 4:
            #     k_fields_dict = MF_Ensemble.get_member_fields(['npf'])
            #     k_fields = [d['npf'] for d in k_fields_dict]
            #     plot_k_fields(gwf, pars,  k_fields)
                
        print('--------')
        print(f'time step {t_step}')
        start_time = time.time()
        rch_data, wel_data, riv_data, Y_obs = get_transient_data(pars, t_step, true_h[t_step], obs_cid)
        MF_Ensemble.update_transient_data(rch_data, wel_data, riv_data)
        print(f'transient data loaded and applied in {(time.time() - start_time):.2f} seconds')
        
        print('---')
        start_time = time.time()
        MF_Ensemble.propagate()
        MF_Ensemble.model_error(true_h[t_step])
        MF_Ensemble.record_state(pars, pars['EnKF_p'])
        print(f'ensemble propagated in {(time.time() - start_time):.2f} seconds')
 
        if Assimilate:
            # print('---')
            start_time = time.time()
            EnKF.update_X_Y(
                MF_Ensemble.get_Kalman_X_Y(
                    pars['EnKF_p']
                    )
                )
            EnKF.analysis()

            EnKF.Kalman_update(Y_obs)

            print(f'Ensemble Kalman Filter performed in  {(time.time() - start_time):.2f} seconds')
            
            start_time = time.time()
            MF_Ensemble.apply_X(pars['EnKF_p'], EnKF.X)

            print(f'Application of results plus kriging took {(time.time() - start_time):.2f} seconds')
            
            start_time = time.time()
            MF_Ensemble.write_simulations()

            print(f'Writing all simulation files took {(time.time() - start_time):.2f} seconds')
        
    
    
    