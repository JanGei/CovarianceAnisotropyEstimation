import numpy as np
import os
import sys
import psutil

def get():
    
    # Changing the working directory to the parent directory to have consisten access
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    Vrdir = parent_directory + '/Virtual_Reality'

    # Cell spacing in x and y
    dx          = np.array([50, 50])
    # Different covariance models
    cov_mods    = ['Exponential', 'Matern', 'Gaussian']
    # Well locations in Erdal & Cirpka
    row_well    = 5
    col_well    = 9
    well_loc    = np.zeros((col_well*row_well,2))
    for i in range(row_well):
        for j in range(col_well):
            well_loc[i*col_well + j, 0] = (19.5 + 10*j) *dx[0] 
            well_loc[i*col_well + j, 1] = (8.5 + 10*i) *dx[1]
    # pumping wells should be at (5, 9, 15, 27, 31)
    q_idx       = [5, 9, 15, 27, 31]
    mask        = np.full(len(well_loc),True,dtype=bool)
    mask[q_idx] = False
    # Model units
    lenuni      = 'METERS'
    timuni      = 'SECONDS'
    
    sim_ws      = os.path.join(Vrdir, 'model_files')
    gg_ws       = os.path.join(Vrdir, 'gridgen_files')
    trs_ws      = os.path.join(Vrdir, 'transient_model')
    vr_h_dir    = os.path.join(Vrdir, 'model_data', 'head_ref.npy')
    vr_obs_dir  = os.path.join(Vrdir, 'model_data', 'obs_ref.npy')
    
    ensemb_dir  = os.path.join(parent_directory, 'Ensemble')
    output_dir  = os.path.join(parent_directory, 'output')
    
    temp_m_ws   = os.path.join(ensemb_dir, 'template_model')
    member_ws   = os.path.join(ensemb_dir, 'member')

    # load reference data
    k_ref_dir   = os.path.join(Vrdir, 'model_data','logK_ref.csv')
    r_ref_dir   = os.path.join(Vrdir, 'model_data','rech_ref.csv')
    rivh_dir   = os.path.join(Vrdir, 'model_data','tssl.csv')
    sfac_dir   = os.path.join(Vrdir, 'model_data','sfac.csv')
    # k_ref       = np.loadtxt(os.path.join(Vrdir, 'model_data','logK_ref.csv'),
    #                          delimiter = ',')
    # r_ref       = np.loadtxt(os.path.join(Vrdir, 'model_data','rech_ref.csv'),
    #                          delimiter = ',')
    # rivh        = np.genfromtxt(os.path.join(Vrdir, 'model_data','tssl.csv'),
    #                             delimiter = ',',
    #                             names=True)['Wert']
    # sfac        = np.genfromtxt(os.path.join(Vrdir, 'model_data','sfac.csv'),
    #                             delimiter = ',',
    #                             names=True)['Wert']
    
    computer = ['office', 'icluster', 'binnac']
    setup = computer[0]
    if setup == 'office':
        n_mem  = 2
        nprocs = np.min([n_mem, psutil.cpu_count()])
        if n_mem == 2:
            nprocs = 1
        up_temp = False
        n_pre_run = 1
    elif setup == 'icluster':
        n_mem  = 120
        nprocs = psutil.cpu_count()
        up_temp = True
        n_pre_run = 40
    elif setup == 'binnac':
        n_mem  = 560
        nprocs = psutil.cpu_count()
        up_temp = True
        n_pre_run = 40
    
    
    choice = 0
    variants = [['cov_data', 'npf'], ['cov_data'], ['npf']]
    pp_flag = True
    if not pp_flag and choice == 0 | choice == 1:
        print("You cant have a variogram with no pilotpoints - yet")
        sys.exit() 
        
        
    if choice == 0:
        covtype = "random"
        valtype = "random"
        
    elif choice == 1:
        covtype = "random"
        valtype = "good"
        
    elif choice == 2:
        covtype = "good"
        if pp_flag:
            valtype = "random"
        else:
            valtype = "random"
            
    h_damp = 0.5
    cov_damp = 0.1
    npf_damp = 1
    l_red = 1
    damp = [[h_damp, cov_damp, npf_damp], [h_damp, cov_damp], [h_damp, npf_damp]]
    
    pars    = {
        'pilotp': pp_flag,
        'nprocs': nprocs,
        'setup' : setup,
        'EnKF_p': variants[choice], 
        'damp'  : damp[choice],
        'n_PP'  : 50,
        'eps'   : 0.05,
        'covt'  : covtype,
        'valt'  : valtype,
        'l_red' : l_red,
        'up_tem': up_temp,
        'nx'    : np.array([100, 50]),                      # number of cells
        'dx'    : dx,                                       # cell size
        'lx'    : np.array([[2000,600], [5000,500]])/l_red, # corellation lengths
        'ang'   : np.array([291, 17]),                      # angle in ° (logK, recharge)
        'sigma' : np.array([1.7, 0.1]),                     # variance (logK, recharge)
        'mu'    : np.array([-8.5, -0.7]),                   # mean (log(ms-1), (mm/d))
        'cov'   : cov_mods[1],                              # Covariance models
        'nlay'  : np.array([1]),                            # Number of layers
        'bot'   : np.array([0]),                            # Bottom of aquifer
        'top'   : np.array([50]),                           # Top of aquifer
        'welxy' : np.array(well_loc[q_idx]),                # location of pumps
        'obsxy' : np.array(well_loc[mask]),                 # location of obs
        'welq'  : np.array([9, 18, 90, 0.09, 0.9])/3600,    # Q of wells [m3s-1]
        'welst' : np.array([20, 300, 200, 0, 0]),           # start day of pump
        'welnd' : np.array([150, 365, 360, 370, 300]),      # end day of pump
        'welay' : np.array(np.zeros(5)),                    # layer of wells
        'river' : np.array([[0.0,0], [5000,0]]),            # start / end of river
        'rivC'  : 5*1e-4,                                   # river conductance [ms-1]
        'rivd'  : 2,                                        # depth of river [m]
        'chd'   : np.array([[0.0,2500], [5000,2500]]),      # start / end of river
        'chdh'  : 15,                                       # initial stage of riv
        'ss'    : 1e-5,                                     # specific storage
        'sy'    : 0.15,                                     # specific yield
        'mname' : "Reference",
        'sname' : "Reference",
        'sim_ws': sim_ws,
        'vr_h_d': vr_h_dir,
        'vr_o_d': vr_obs_dir,
        'gg_ws' : gg_ws,
        'ens_ws': ensemb_dir,
        'mem_ws': member_ws,
        'timuni': timuni,                                   # time unit
        'lenuni': lenuni,                                   # length unit
        'k_r_d' : k_ref_dir,
        # 'kmin'  : np.min(np.log(k_ref)),
        # 'kmax'  : np.max(np.log(k_ref)),
        'r_r_d' : r_ref_dir,
        'rh_d'  : rivh_dir,
        'sf_d'  : sfac_dir,
        'n_mem' : n_mem,
        'tm_ws' : temp_m_ws,
        'trs_ws': trs_ws ,
        'resdir': output_dir,
        'nsteps': int(365*24/6),
        'nprern': n_pre_run
        }
    
    return pars
