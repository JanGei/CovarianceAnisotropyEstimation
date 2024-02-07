from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from dependencies.convert_transient import convert_to_transient
from dependencies.create_pilot_points import create_pilot_points
from dependencies.create_k_fields import create_k_fields
from dependencies.load_template_model import load_template_model
from dependencies.plot import plot_fields, plot_POI, plot_k_fields
from objects.Ensemble import Ensemble
from objects.Member import Member
import numpy as np


if __name__ == '__main__':
    pars        = get()
    # n_mem       = pars['n_mem']
    tm_ws       = pars['tm_ws']
    # ss_mod      = pars['sim_ws']
    # sname       = pars['sname']
    # mname       = pars['mname']
    upd_temp    = pars['up_tem']
    n_mem       = pars['n_mem']
    # n_PP        = pars['n_PP']
    
    if upd_temp:
        temp_sim = convert_to_transient(tm_ws, pars)
    
    #function to load base model
    model_dir   = create_Ensemble(pars)
    sim, gwf = load_template_model(pars)
    
    
    pp_cid, pp_xy = create_pilot_points(gwf, pars)
    # plot_POI(gwf, pp_xy, pars)
    # HIER MÜSSEN WIR NOCH DIE Covarianzmodelle mitgeben oder zumindest die Parameter
    k_fields = create_k_fields(gwf, pars, pp_xy, pp_cid, covtype = 'random')
    # plot_fields(gwf, pars,  k_fields[0], k_fields[1])
    plot_k_fields(gwf, pars,  k_fields)
    
    # Wir haben jetzt Felder mit random covarianz strukturen --> lets pack them into members
    