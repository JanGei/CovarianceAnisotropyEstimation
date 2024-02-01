from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from dependencies.convert_transient import convert_to_transient
from dependencies.create_pilot_points import create_pilot_points
from objects.Ensemble import Ensemble
from objects.Member import Member


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
    
    model_dir = create_Ensemble(pars)
    
    # pilot_p = create_pilot_points(pars)
    # k_fields = create_k_fields(pars)
    