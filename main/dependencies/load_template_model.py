import flopy

def load_template_model(pars: dict,  SS = False):
    
    if SS:
        temp_m_ws = pars['ss_ws']
    else:
        temp_m_ws = pars['trs_ws']
    mname = pars['mname']
    
    sim        = flopy.mf6.modflow.MFSimulation.load(
                            version             = 'mf6', 
                            exe_name            = 'mf6',
                            sim_ws              = temp_m_ws, 
                            verbosity_level     = 0
                            )
    gwf = sim.get_model(mname)
    
    return sim, gwf