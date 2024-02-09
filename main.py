from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from dependencies.convert_transient import convert_to_transient
from dependencies.create_pilot_points import create_pilot_points
from dependencies.create_k_fields import create_k_fields
from dependencies.load_template_model import load_template_model
from dependencies.load_observations import load_observations
from dependencies.get_transient_data import get_transient_data
from dependencies.plot import plot_fields, plot_POI, plot_k_fields, plot
from dependencies.intersect_with_grid import intersect_with_grid
from objects.Ensemble import Ensemble
from objects.MFModel import MFModel
from objects.EnsembleKalmanFilter import EnsembleKalmanFilter
import numpy as np
from joblib import Parallel, delayed
import warnings

# Suppress DeprecationWarning temporarily
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

if __name__ == '__main__':
    pars        = get()
    n_mem       = pars['n_mem']

    if pars['office']:
        nprocs = np.min([n_mem, 6])
        # nprocs = 1
    elif pars['iclust']:
        nprocs = 12
    elif pars['binnac']:
        nprocs = 12
    
    if pars['up_tem']:
        temp_sim = convert_to_transient(pars['tm_ws'], pars)
    
    # copy template model to ene
    model_dir   = create_Ensemble(pars)
    sim, gwf = load_template_model(pars)
    
    
    pp_cid, pp_xy = create_pilot_points(gwf, pars)
    obs_cid = intersect_with_grid(gwf, pars['obsxy'])
    obs_val = load_observations(pars)
    # plot_POI(gwf, pp_xy, pars, bc = True)

    k_fields, cov_data = create_k_fields(gwf, pars, pp_xy, pp_cid, covtype = 'random', valtype = 'random')
    # plot_fields(gwf, pars,  k_fields[0], k_fields[1])
    # plot_k_fields(gwf, pars,  k_fields)
    # plot(gwf, ['logK','h'], bc=True)
    
    # generate model instances  
    models = Parallel(n_jobs=nprocs)(delayed(MFModel)(
        model_dir[idx],
        pars['mname'],
        cov_data[idx]) 
        for idx in range(n_mem)
        )
    
    # add the models to the ensemble
    MF_Ensemble     = Ensemble(models, nprocs, pp_cid, pp_xy, obs_cid)
    
    # set their respective k-fields
    MF_Ensemble.set_field(k_fields, ['npf'])
    
    # Running each model 10 times, setting the results as initial conidition to
    # improve initial accuracy
    for idx in range(10):
        MF_Ensemble.propagate()
        MF_Ensemble.update_initial_heads()
        
    X, Ysim = MF_Ensemble.get_Kalman_X_Y(['cov_data', 'npf'])
    EnKF = EnsembleKalmanFilter(X, Ysim, damp = 0.75, eps = 0.05)
    
    # for t_step in range(pars['nsteps']):
    for t_step in range(2):
        
        rch_data, wel_data, riv_data, Y_obs = get_transient_data(pars, t_step, obs_val)
        MF_Ensemble.update_transient_data(rch_data, wel_data, riv_data)
        
        MF_Ensemble.propagate()
        
        EnKF.analysis()
        # X = EnKF.Kalman_update(Y_obs)
        
        MF_Ensemble.apply_X(['cov_data', 'npf'],X)



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    