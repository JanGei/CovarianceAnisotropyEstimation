import numpy as np

def get():
    
    cov_mods = ['Exponential', 'Matern', 'Gaussian']
    
    pars    = {
        'nx'    : np.array([1000, 500]),                    # number of cells
        'dx'    : np.array([50, 50]),                       # cell size
        'lx'    : np.array([[600, 2000], [500, 5000]]),     # corellation lengths
        'ang'   : np.array([291, 17])+90,                   # angle (logK, recharge)
        'sigma' : np.array([1.7, 0.1]),                     # variance (logK, recharge)
        'mu'    : np.array([-8.5, -0.7]),                   # mean (logK, recharge)
        'cov'   : cov_mods[0]                               # Covariance models
        }

    return pars
