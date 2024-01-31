from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from dependencies.convert_transient import convert_to_transient


if __name__ == '__main__':
    pars        = get()
    # n_mem       = pars['n_mem']
    tm_ws       = pars['tm_ws']
    # ss_mod      = pars['sim_ws']
    # sname       = pars['sname']
    # mname       = pars['mname']
    upd_temp    = pars['up_tem']
    
    if upd_temp:
        temp_sim = convert_to_transient(tm_ws, pars)
    
    