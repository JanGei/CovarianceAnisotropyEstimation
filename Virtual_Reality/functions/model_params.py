import numpy as np

def get():
    
    dx          = np.array([50, 50])
    cov_mods    = ['Exponential', 'Matern', 'Gaussian']
    
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

    pars    = {
        'nx'    : np.array([100, 50]),                      # number of cells
        'dx'    : dx,                                       # cell size
        'lx'    : np.array([[600, 2000], [500, 5000]]),     # corellation lengths
        'ang'   : np.array([291, 17])+90,                   # angle (logK, recharge)
        'sigma' : np.array([1.7, 0.1]),                     # variance (logK, recharge)
        'mu'    : np.array([-8.5, -0.7]),                   # mean (log(ms-1), (mmd-1))
        'cov'   : cov_mods[0],                              # Covariance models
        'nlay'  : np.array([1]),                            # Number of layers
        'bot'   : np.array([0]),                            # Bottom of aquifer
        'top'   : np.array([50]),                           # Top of aquifer
        'welxy' : np.array(well_loc[q_idx]),                # location of pumps
        'obsxy' : np.array(well_loc[mask]),                 # location of obs
        'welq'  : np.array([9, 18, 90, 0.09, 0.9]),         # Q of wells [m3h-1]
        'welay' : np.array(np.zeros(5)),                    # layer of wells
        'river' : np.array([[0.0,0], [1000.0,0]]),          # start / end of river
        'rivh'  : 13.4,                                     # initial stage of riv
        'rivC'  : abs(1e-5*86400),                          # river conductance [md-1]
        }

    return pars
