import numpy as np

def get():
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
            well_loc[i*col_well + j, 0] = (20 + 10*j) *dx[0]
            well_loc[i*col_well + j, 1] = (10 + 10*i) *dx[1]
    # pumping wells should be at (5, 9, 15, 27, 31)
    q_idx       = [5, 9, 15, 27, 31]
    mask        = np.full(len(well_loc),True,dtype=bool)
    mask[q_idx] = False
    # Model units
    lenuni      = 'METERS'
    timuni      = 'SECONDS'

    pars    = {
        'nx'    : np.array([100, 50]),                      # number of cells
        'dx'    : dx,                                       # cell size
        'lx'    : np.array([[600, 2000], [500, 5000]]),     # corellation lengths
        'ang'   : np.array([291, 17])+90,                   # angle (logK, recharge)
        'sigma' : np.array([1.7, 0.1]),                     # variance (logK, recharge)
        'mu'    : np.array([-8.5, -0.7]),                   # mean (log(ms-1), (mm/d))
        'cov'   : cov_mods[0],                              # Covariance models
        'nlay'  : np.array([1]),                            # Number of layers
        'bot'   : np.array([0]),                            # Bottom of aquifer
        'top'   : np.array([50]),                           # Top of aquifer
        'welxy' : np.array(well_loc[q_idx]),                # location of pumps
        'obsxy' : np.array(well_loc[mask]),                 # location of obs
        'welq'  : np.array([9, 18, 90, 0.09, 0.9])/3600,    # Q of wells [m3s-1]
        'welay' : np.array(np.zeros(5)),                    # layer of wells
        'river' : np.array([[0.0,0], [5000,0]]),            # start / end of river
        'rivC'  : 1e-5,                                     # river conductance [ms-1]
        'chd'   : np.array([[0.0,2500], [5000,2500]]),      # start / end of river
        'chdh'  : 15,                                       # initial stage of riv
        'ss'    : 1e-5,                                     # specific storage
        'sy'    : 0.15,                                     # specific yield
        'mname' : "Reference",
        'sname' : "Reference",
        'sim_ws': "./model_files",
        'timuni': timuni,                                   # time unit
        'lenuni': lenuni,                                   # length unit
        'k_ref' : np.loadtxt('model_data/logK_ref.csv',
                             delimiter = ','),
        'r_ref' : np.loadtxt('model_data/rech_ref.csv',
                             delimiter = ','),
        'rivh'  : np.genfromtxt('model_data/tssl.csv',
                                delimiter = ',',
                                names=True)['Wert'],
        'sfac'  : np.genfromtxt('model_data/sfac.csv',
                                delimiter = ',',
                                names=True)['Wert'],
        }

    return pars
