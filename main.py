from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from dependencies.convert_transient import convert_to_transient
from dependencies.create_pilot_points import create_pilot_points
from dependencies.create_k_fields import create_k_fields
from dependencies.load_template_model import load_template_model
from dependencies.plot import plot_fields, plot_POI, plot_k_fields
from objects.Ensemble import Ensemble
from objects.MFModel import MFModel
import numpy as np
from joblib import Parallel, delayed
import warnings

# Suppress DeprecationWarning temporarily
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
if __name__ == '__main__':
    pars        = get()
    # n_mem       = pars['n_mem']
    tm_ws       = pars['tm_ws']
    # ss_mod      = pars['sim_ws']
    # sname       = pars['sname']
    mname       = pars['mname']
    upd_temp    = pars['up_tem']
    n_mem       = pars['n_mem']
    # n_PP        = pars['n_PP']
    if pars['office']:
        nprocs = np.min([n_mem, 6])
        # nprocs = 1
    elif pars['iclust']:
        nprocs = 12
    elif pars['binnac']:
        nprocs = 12
    
    if upd_temp:
        temp_sim = convert_to_transient(tm_ws, pars)
    
    # copy template model to ene
    model_dir   = create_Ensemble(pars)
    sim, gwf = load_template_model(pars)
    
    
    pp_cid, pp_xy = create_pilot_points(gwf, pars)
    # plot_POI(gwf, pp_xy, pars)

    k_fields, cov_data = create_k_fields(gwf, pars, pp_xy, pp_cid, covtype = 'random', valtype = 'random')
    # plot_fields(gwf, pars,  k_fields[0], k_fields[1])
    plot_k_fields(gwf, pars,  k_fields)
    
    # generate model instances  
    models = Parallel(n_jobs=nprocs)(delayed(MFModel)(
        model_dir[idx],
        mname,
        cov_data[idx]) 
        for idx in range(n_mem)
        )
    
    # add the models to the ensemble
    MF_Ensemble     = Ensemble(models, nprocs, pp_cid, pp_xy)
    
    # set their respective k-fields
    MF_Ensemble.set_field(k_fields, ['npf'])
    
    # Running each model 10 times, setting the results as initial conidition to
    # improve initial accuracy
    for idx in range(10):
        MF_Ensemble.propagate()
        MF_Ensemble.update_initial_heads()
        
    X = MF_Ensemble.get_Kalman_X(['cov_data', 'npf'])
    
    # get this one going
    # Ysim = MF_Ensemble.get_Kalman_Ysim()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    