from dependencies.model_params import get
from dependencies.copy import create_Ensemble
# from dependencies.convert_transient import convert_to_transient
from dependencies.create_pilot_points import create_pilot_points_even, create_pilot_points
from dependencies.load_template_model import load_template_model
from dependencies.create_k_fields import create_k_fields
from dependencies.write_file import write_file
from dependencies.get_transient_data import get_transient_data
from dependencies.intersect_with_grid import intersect_with_grid
from dependencies.generate_mask import chd_mask
from dependencies.plotting.ellipses import ellipses
from dependencies.plotting.compare_mean import compare_mean_true
from dependencies.plotting.check_observations import check_observations
# from dependencies.plotting.plot_k_fields import plot_k_fields
from dependencies.plotting.plot_k_fields import plot_k_fields
from objects.Ensemble import Ensemble
from objects.MFModel import MFModel
from objects.Virtual_Reality import Virtual_Reality
from objects.EnsembleKalmanFilter import EnsembleKalmanFilter
from Virtual_Reality.ReferenceModel import create_reference_model
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
        create_reference_model(pars)
        
    VR_Model = Virtual_Reality(pars)
    
    print(f'Joblib initiated with {nprocs} processors')
    print(f'The template model is located in {pars["tm_ws"]}')
    #%% loading necessary data
    start_time = time.time()
    
    # copy template model to ensemble folder
    model_dir = create_Ensemble(pars)
    sim, gwf = load_template_model(pars)
    
    obs_cid = intersect_with_grid(gwf, pars['obsxy'])
    
    k_fields = []
    cor_ellips = []
    l_angs = []
    pp_k_ini = []
    
    
    if pars['pilotp']:
        if pars['ppeven']:
            pp_cid, pp_xy, near_dist = create_pilot_points_even(gwf, pars)
        else:
            pp_cid, pp_xy, near_dist = create_pilot_points(gwf, pars)
            
        write_file(pars,[pp_cid, pp_xy], ["pp_cid","pp_xy"], 0, intf = True)
        # create_k_fields
        result = Parallel(n_jobs=nprocs, backend = "threading")(delayed(create_k_fields)(
            gwf,
            pars, 
            pp_xy,
            pp_cid,
            conditional = pars['condfl']
            )
            for idx in range(n_mem)
            )
        # sorting the results
        for tup in result:
            field, ellips, l_ang, pilotpoints = tup
            k_fields.append(field)
            cor_ellips.append(ellips)
            l_angs.append(l_ang)
            pp_k_ini.append(pilotpoints[1])
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
                               pp_xy,
                               pp_k_ini)
    
    m = np.mean(cor_ellips, axis = 0)
    mat = np.array([[m[0], m[1]],[m[1], m[2]]])
    ellipses(
        MF_Ensemble.ellipses,
        pars['mat2cv'](mat),
        pars
        )
    # set their respective k-fields
    MF_Ensemble.set_field(k_fields, ['npf'])
    # MF_Ensemble.set_field([VR_Model.npf.k.array for i in range(len(models))], ['npf'])
    
    if pars['printf']: print(f'Ensemble is initiated and respective k-fields are set in {(time.time() - start_time):.2f} seconds')
    
    #%%
    X, Ysim, _ = MF_Ensemble.get_Kalman_X_Y()
    damp = MF_Ensemble.get_damp(X, pars)
    EnKF = EnsembleKalmanFilter(X, Ysim, damp = damp, eps = pars['eps'])
    true_h = np.zeros((pars['nsteps'],len(VR_Model.cxy)))
    mean_h = np.zeros((pars['nsteps'],len(VR_Model.cxy)))
    var_h = np.zeros((pars['nsteps'],len(VR_Model.cxy)))
    true_obs = np.zeros((pars['nsteps'],len(obs_cid)))
    mean_obs = np.zeros((pars['nsteps'],len(obs_cid)))
    MF_Ensemble.remove_current_files(pars)

    # for t_step in range(pars['nsteps']):
    for t_step in range(pars['nsteps']):
        
        period, Assimilate = pars['period'](t_step, pars)  
        if t_step/4 == pars['asim_d'][1]:
            MF_Ensemble.reset_errors()
        print('--------')
        print(f'time step {t_step}')
        start_time_ts = time.time()
        if t_step%4 == 0:
            rch_data, wel_data, riv_data = get_transient_data(pars, t_step)

            MF_Ensemble.update_transient_data(rch_data, wel_data, riv_data)
            VR_Model.update_transient_data(rch_data, wel_data, riv_data)

            if pars['printf']: print(f'transient data loaded and applied in {(time.time() - start_time_ts):.2f} seconds')
        
        if pars['printf']: print('---')
        start_time = time.time()
        VR_Model.simulation()
        MF_Ensemble.propagate()
        
        if pars['printf']: print(f'Ensemble propagated in {(time.time() - start_time):.2f} seconds')
 
        if Assimilate:
            # print('---')
            start_time = time.time()
            X, Ysim, mean_obs[t_step,:] = MF_Ensemble.get_Kalman_X_Y()
            EnKF.update_X_Y(X, Ysim)
            EnKF.analysis()
            true_obs[t_step,:] = np.squeeze(VR_Model.get_observations(obs_cid))
            EnKF.Kalman_update(true_obs[t_step,:].T)

            if pars['printf']: print(f'Ensemble Kalman Filter performed in  {(time.time() - start_time):.2f} seconds')

            start_time = time.time()
            MF_Ensemble.apply_X(EnKF.X, pars['EnKF_p'])

            if pars['printf']: print(f'Application of results plus kriging took {(time.time() - start_time):.2f} seconds')
        else:
            # Very important: update initial conditions if youre not assimilating
            MF_Ensemble.update_initial_heads()
        
        true_h[t_step,:] = VR_Model.update_ic()
        mean_h[t_step,:], var_h[t_step,:] = MF_Ensemble.get_mean_var(h = 'ic')

        start_time = time.time()
        if period == "assimilation" or period == "prediction":
            if t_step%4 == 0:
                MF_Ensemble.model_error(true_h[t_step], mean_h[t_step], var_h[t_step], period)
                MF_Ensemble.record_state(pars, pars['EnKF_p'], true_h[t_step], period)
            
                # visualize covariance structures
                if pars['setup'] == 'office' and Assimilate and t_step%20 == 0:
                    if 'cov_data' in pars['EnKF_p']:
                        m = MF_Ensemble.mean_cov_par
                        mat = np.array([[m[0], m[1]],[m[1], m[2]]])
                        ellipses(
                            MF_Ensemble.ellipses,
                            pars['mat2cv'](mat),
                            pars
                            )
                    if t_step%40 == 60:
                        compare_mean_true(gwf, [np.squeeze(VR_Model.npf.k.array), MF_Ensemble.meanlogk]) 
                
                if pars['printf']: print(f'Plotting and recording took {(time.time() - start_time):.2f} seconds')
                if pars['printf']: print(f'Entire Step took {(time.time() - start_time_ts):.2f} seconds')
    