from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from dependencies.convert_transient import convert_to_transient
from dependencies.create_pilot_points import create_pilot_points
from dependencies.load_template_model import load_template_model
from dependencies.create_k_fields import create_k_fields
from dependencies.write_file import write_file
from dependencies.load_observations import load_true_h_field
from dependencies.get_transient_data import get_transient_data
from dependencies.intersect_with_grid import intersect_with_grid
from dependencies.generate_mask import chd_mask
from dependencies.plotting.ellipses import ellipses
from dependencies.plotting.compare_mean import compare_mean_true
# from dependencies.plotting.plot_k_fields import plot_k_fields
from dependencies.plotting.plot_k_fields import plot_k_fields
from objects.Ensemble import Ensemble
from objects.MFModel import MFModel
from objects.EnsembleKalmanFilter import EnsembleKalmanFilter
from Virtual_Reality.ReferenceModel import run_reference_model
from Virtual_Reality.functions.generator import gsgenerator
import time
import numpy as np
import os
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
        run_reference_model(pars)
        temp_sim = convert_to_transient(pars['tm_ws'], pars)
        
    
    print(f'Joblib initiated with {nprocs} processors')
    print(f'The template model is located in {pars["tm_ws"]}')
    #%% loading necessary data
    start_time = time.time()
    
    # copy template model to ensemble folder
    model_dir = create_Ensemble(pars)
    sim, gwf = load_template_model(pars)
    
    obs_cid = intersect_with_grid(gwf, pars['obsxy'])
    # not needed here, but will be needed in non synthetic cases
    # obs_val = load_observations(pars) 
    true_h = load_true_h_field(pars)
    
    k_fields = []
    cor_ellips = []
    l_angs = []
    
    
    if pars['pilotp']:
        pp_cid, pp_xy, near_dist = create_pilot_points(gwf, pars)
        write_file(pars,[pp_cid, pp_xy], ["pp_cid","pp_xy"], 0, intf = True)
        # create_k_fields
        result = Parallel(n_jobs=nprocs, backend = "threading")(delayed(create_k_fields)(
            gwf,
            pars, 
            pp_xy,
            pp_cid,
            )
            for idx in range(n_mem)
            )
        # sorting the results
        for tup in result:
            field, ellips, l_ang, _ = tup
            k_fields.append(field)
            cor_ellips.append(ellips)
            l_angs.append(l_ang)
    else:
        k_fields = Parallel(n_jobs=nprocs, backend = "threading")(delayed(gsgenerator)(
            gwf,
            pars,
            pars['lx'][0], 
            pars['ang'][0],
            pars['sigma'][0],
            pars['cov'],
            pars['mu'],
            covtype = pars['covt'],
            valtype = pars['valt']) 
            for idx in range(n_mem)
            )
        for field in k_fields:
            cor_ellips.append([])
            l_angs.append([])
            pp_xy, pp_cid = [], []
        
    # save original fields
    if pars['setup'] == 'binnac':
        np.save(os.path.join(pars['resdir'] ,'k_ensemble_ini.npy'), k_fields)
    k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')
    
    # plot_POI(gwf, pp_xy, pars, bc = True)
    # plot_fields(gwf, pars,  k_fields[0], k_fields[1])
    # plot_k_fields(gwf, pars,  k_fields)
    
    # plot(gwf, ['logK','h'], bc=True)
    
    mask_chd = chd_mask(gwf)
    
    print(f'The model has {len(obs_cid)} observation points')
    if pars['pilotp']:
        print(f'The model has {len(pp_cid)} pilot points points')
    if pars['printf']: print(f'Loading of data and creating k_fields took {(time.time() - start_time):.2f} seconds')
    
    #%% generate model instances  
    start_time = time.time()
    models = Parallel(n_jobs=nprocs, backend="threading")(delayed(MFModel)(
        model_dir[idx],
        pars,
        near_dist,
        l_angs[idx],
        cor_ellips[idx]) 
        for idx in range(n_mem)
        )
    
    if pars['printf']: print(f'{n_mem} models are initiated in {(time.time() - start_time):.2f} seconds')
    #%% add the models to the ensemble
    start_time = time.time()
    
    MF_Ensemble     = Ensemble(models,
                               pars['pilotp'],
                               nprocs,
                               obs_cid,
                               mask_chd,
                               np.array(l_angs),
                               np.array(cor_ellips),
                               pp_cid,
                               pp_xy)
    
    # set their respective k-fields
    MF_Ensemble.set_field(k_fields, ['npf'])
    # plot_k_fields(gwf, pars,  k_fields, np.rad2deg(MF_Ensemble.ellipses[:,2]))
    if pars['printf']: print(f'Ensemble is initiated and respective k-fields are set in {(time.time() - start_time):.2f} seconds')
    #%% Running each model n times
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
    for t_step in range(pars['nsteps']):
        if t_step == 0:
            MF_Ensemble.remove_current_files(pars)
        if t_step == 2750:
            Assimilate = False
                
        print('--------')
        print(f'time step {t_step}')
        start_time = time.time()
        rch_data, wel_data, riv_data, Y_obs = get_transient_data(pars, t_step, true_h[t_step], obs_cid)
        start_time = time.time()
        MF_Ensemble.update_transient_data(rch_data, wel_data, riv_data)
        if pars['printf']: print(f'transient data loaded and applied in {(time.time() - start_time):.2f} seconds')
        
        if pars['printf']: print('---')
        start_time = time.time()
        MF_Ensemble.propagate()
        if pars['printf']: print(f'Ensemble propagated in {(time.time() - start_time):.2f} seconds')
 
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

            if pars['printf']: print(f'Ensemble Kalman Filter performed in  {(time.time() - start_time):.2f} seconds')
            
            start_time = time.time()
            MF_Ensemble.apply_X(pars['EnKF_p'], EnKF.X)

            if pars['printf']: print(f'Application of results plus kriging took {(time.time() - start_time):.2f} seconds')
        else:
            # Very important: update initial conditions if youre not assimilating
            MF_Ensemble.update_initial_heads()

        start_time = time.time()
        MF_Ensemble.model_error(true_h[t_step])
        MF_Ensemble.record_state(pars, pars['EnKF_p'])
        # visualize covariance structures
        if pars['setup'] == 'office' and t_step%10 == 0:
            if 'cov_data' in pars['EnKF_p']:
                eigenvalues, eigenvectors = np.linalg.eig(MF_Ensemble.mean_cov_par)
                ellipses(
                    MF_Ensemble.ellipses,
                    pars['mat2cv'](eigenvalues, eigenvectors),
                    pars
                    )
        
            
            if t_step%50 == 0:
                k_fields = MF_Ensemble.get_member_fields(['npf'])
                plot_k_fields(gwf, pars,  [field['npf'] for field in k_fields[0:8]])
                compare_mean_true(gwf, [k_ref, MF_Ensemble.logmeank]) 
            
        if pars['printf']: print(f'Plotting and recording took {(time.time() - start_time):.2f} seconds')
    
    