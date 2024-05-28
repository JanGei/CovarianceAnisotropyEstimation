import numpy as np
import os
import sys
import psutil

def create_wells(row_well, col_well, dx):
    well_loc = np.zeros((col_well*row_well,2))
    for i in range(row_well):
        for j in range(col_well):
            well_loc[i*col_well + j, 0] = (19.5 + 10*j) *dx[0] 
            well_loc[i*col_well + j, 1] = (8.5 + 10*i) *dx[1]
    # pumping wells should be at (5, 9, 15, 27, 31)
    return well_loc

def covariance_matrix(H, sigma2, Ctype):
    if Ctype == 'Exponential':
        covmat = sigma2 * np.exp(-H)
    elif Ctype == 'Gaussian':
        covmat = sigma2 * np.exp(-H ** 2)
    elif Ctype == 'Matern':
        covmat = sigma2 * np.multiply((1+np.sqrt(3)*H), np.exp(-np.sqrt(3)*H))
        
    return covmat

def rotation_matrix(angle):
    # This formulation rotates counter-clockwise from x-axis
    # To rotate clockwise, you need the inverse of this rotation matrix, i.e.
    # flipping the signs of the sines
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) 

def rotate2Dfield(X,Y, angle):
    rotmat = rotation_matrix(angle)
    Xrot = X * rotmat[0,0] + Y * rotmat[0,1]
    Yrot = X * rotmat[1,0] + Y * rotmat[1,1]
    
    return Xrot, Yrot
def extract_truth(eigenvalues, eigenvectors):
     
     lxmat = 1/np.sqrt(eigenvalues)
     
     if lxmat[0] < lxmat[1]:
         lxmat = np.flip(lxmat)
         eigenvectors = np.flip(eigenvectors, axis = 1)
     
     if eigenvectors[0,0] > 0:
         ang = np.pi/2 -np.arccos(np.dot(eigenvectors[:,0],np.array([0,1])))    

     else:
         if eigenvectors[1,0] > 0:
             ang = np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))

         else:
             ang = np.pi -np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))

     return lxmat[0], lxmat[1], ang

def get():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    Vrdir = os.path.join(parent_directory, 'Virtual_Reality')
    ensemb_dir  = os.path.join(parent_directory, 'Ensemble')

    dx          = np.array([50, 50])
    row_well    = 5
    col_well    = 9
    well_loc    = create_wells(row_well, col_well, dx)
    
    q_idx       = [5, 9, 15, 27, 31]
    mask        = np.full(len(well_loc),True,dtype=bool)
    mask[q_idx] = False
    
    cov_mods    = ['Exponential', 'Matern', 'Gaussian']
    computer = ['office', 'icluster', 'binnac']
    setup = computer[0]
    if setup == 'office':
        n_mem  = 32
        nprocs = np.min([n_mem, psutil.cpu_count()])
        if n_mem == 2:
            nprocs = 1
        up_temp = False
        n_pre_run = 1
        printf = True
    elif setup == 'icluster':
        n_mem  = 120
        nprocs = psutil.cpu_count()
        up_temp = True
        n_pre_run = 20
        printf = False
    elif setup == 'binnac':
        n_mem  = 280
        nprocs = psutil.cpu_count()
        up_temp = True
        n_pre_run = 20
        printf = False
    
    choice = [1, 0]
    cov_variants = [['cov_data', 'npf'], ['cov_data'], ['npf']]
    est_variants = ["underestimate", "good", "overestimate"]
    pp_flag = True
    l_red = 5 # possible are 5 and 10
    nPP = 50
    
    h_damp = 0.5
    cov_damp = 0.1
    npf_damp = 0.2
    damp = [[h_damp, cov_damp, npf_damp], [h_damp, cov_damp], [h_damp, npf_damp]]
    
    
    if choice[0] == 0:
        covtype = "random"
        valtype = "random"
        
    elif choice[0] == 1:
        covtype = "random"
        valtype = "good"
        
    elif choice[0] == 2:
        covtype = "random"
        if pp_flag:
            valtype = "random"
        else:
            valtype = "random"

    
    pars    = {
        'pilotp': pp_flag,
        'nprocs': nprocs,
        'setup' : setup,
        'EnKF_p': cov_variants[choice[0]], 
        'damp'  : damp[choice[0]],
        'estyp' : est_variants[choice[1]],
        'n_PP'  : nPP,
        'eps'   : 0.05,
        'omitc' : 2,
        'sig_me': 0.1,
        'geomea': 1,
        'covt'  : covtype,
        'valt'  : valtype,
        'l_red' : l_red,
        'up_tem': up_temp,
        'nx'    : np.array([100, 50]),                      # number of cells
        'dx'    : dx,                                       # cell size
        'lx'    : np.array([[2000,600], [5000,500]])/l_red, # corellation lengths
        'ang'   : np.array([111, 17]),                      # angle in ° (logK, recharge)
        'sigma' : np.array([1.7, 0.1]),                     # variance (logK, recharge)
        'mu'    : np.array([-8.5, -0.7]),                   # mean (log(ms-1), (mm/d))
        'cov'   : cov_mods[1],                              # Covariance models
        'nlay'  : np.array([1]),                            # Number of layers
        'bot'   : np.array([0]),                            # Bottom of aquifer
        'top'   : np.array([50]),                           # Top of aquifer
        'welxy' : np.array(well_loc[q_idx]),                # location of pumps
        'obsxy' : np.array(well_loc[mask]),                 # location of obs
        'welq'  : np.array([35, 18, 90, 20, 15])/3600,      # Q of wells [m3s-1]
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
        'printf': printf,
        'sim_ws': os.path.join(Vrdir, 'model_files'),
        'vr_h_d': os.path.join(Vrdir, 'model_data', 'head_ref.npy'),
        'vr_o_d': os.path.join(Vrdir, 'model_data', 'obs_ref.npy'),
        'gg_ws' : os.path.join(Vrdir, 'gridgen_files'),
        'ens_ws': ensemb_dir,
        'mem_ws': os.path.join(ensemb_dir, 'member'),
        'timuni': 'SECONDS',                                   # time unit
        'lenuni': 'METERS',                                   # length unit
        'k_r_d' : os.path.join(Vrdir, 'model_data','logK_ref.csv'),
        'r_r_d' : os.path.join(Vrdir, 'model_data','rech_ref.csv'),
        'rh_d'  : os.path.join(Vrdir, 'model_data','tssl.csv'),
        'sf_d'  : os.path.join(Vrdir, 'model_data','sfac.csv'),
        'n_mem' : n_mem,
        'tm_ws' : os.path.join(ensemb_dir, 'template_model'),
        'trs_ws': os.path.join(Vrdir, 'transient_model') ,
        'resdir': os.path.join(parent_directory, 'output'),
        'nsteps': int(2*365*24/6),
        'nprern': n_pre_run,
        'rotmat': rotation_matrix,
        'mat2cv': extract_truth,
        'rot2df': rotate2Dfield,
        'covmat': covariance_matrix,
        }
    
    if choice == 0 or choice == 1:
        if not pp_flag:
            print("You cant have a variogram with no pilotpoints - yet")
            print("Exiting...")
            sys.exit() 
        if l_red < 5:
            print("Your system must not be dominated by the correlation length, if you want to estimate it")
            print("Exiting...")
            sys.exit()
            
    return pars
