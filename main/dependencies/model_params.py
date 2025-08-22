#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This parameter file contains functions and parameters 

@author: janek geiger
"""

import numpy as np
import os
import sys
import psutil

def create_wells(row_well, col_well, dx):
    """
    Creating the pumping wells

    @author: janek geiger
    """
    well_loc = np.zeros((col_well*row_well,2))
    for i in range(row_well):
        for j in range(col_well):
            well_loc[i*col_well + j, 0] = (10 + 10*j) *dx[0] 
            well_loc[i*col_well + j, 1] = (45 - 10*i) *dx[1] 


    # pumping wells should be at (27, 9, 31, 5, 15)
    return well_loc

def distance_matrix(X1,X2,lx=1,ly=1):
    #dstmat
    # calculates the distances between all points in two (n x dim) matrices
    # that are odered pairwise according to their dimension
    X1 = X1 / np.array([lx, ly])
    X2 = X2 / np.array([lx, ly])
    H = np.linalg.norm(X1[:, np.newaxis] - X2, axis=2)
    
    return H

def covariance_matrix(H, sigma2, Ctype):
    #covmat
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
    # HOWEVER, AS WE NEED TO ALIGN A VARIOGRAM WHICH HAS BEEN ROTATED COUNTER-
    # CLOCKWISE AND IS DESTINED TO BE ORIENTED ALONG THE X-AXIS, WE NEED TO 
    # ROTATE THE ENTIRE SYSTEM CLOCKWISE TO COMPENSATE THE ROTATION OF THE
    # VARIOGRAM 
    # ROTATION MATRIX CLOCKWISE
    # cos(a) sin(a)
    # -sin(a) cos(a)
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]]) 

def rotate2Dfield(X,Y, angle):
    # rot2df
    rotmat = rotation_matrix(angle)
    Xrot = X * rotmat[0,0] + Y * rotmat[0,1]
    Yrot = X * rotmat[1,0] + Y * rotmat[1,1]
    
    return Xrot, Yrot


def extract_truth(M):
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Sort the eigenvalues and corresponding eigenvectors in ascending order
    idx = eigenvalues.argsort()  # Indices of sorted eigenvalues in ascending order
    eigenvalues = eigenvalues[idx]  # Sorted eigenvalues
    eigenvectors = eigenvectors[:, idx]  # Corresponding sorted eigenvectors
    
    # Eigenvalues
    lambda1, lambda2 = eigenvalues
    
    # Lengths of the semi-major and semi-minor axes
    lx = 1 / np.sqrt(lambda1)  # lx is the larger length (semi-major axis)
    ly = 1 / np.sqrt(lambda2)  # ly is the smaller length (semi-minor axis)
    
    # Choose the eigenvector corresponding to the larger eigenvalue as the semi-major axis
    v1 = eigenvectors[:, 0] 
    
    # Angle of orientation relative to the semi-major axis
    theta = np.arctan2(v1[1], v1[0])
    
    return lx, ly, np.mod(theta, np.pi)

def period(t_step, pars):
    
    day = np.floor(t_step / 4) + 1 
        
    if day > pars['asim_d'][0]:
        if day > pars['asim_d'][1]:
            period = "prediction"
            Assimilate = False
        else:
            period = "assimilation"
            if t_step%4 == 0:
                Assimilate = True
            else:
                Assimilate = False
    else:
        period = "spinup"
        Assimilate = False
        
    return period, Assimilate


def get():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    Vrdir = os.path.join(parent_directory, 'Virtual_Reality')
    ensemb_dir  = os.path.join(parent_directory, 'Ensemble')
    ghost_ensemb_dir  = os.path.join(parent_directory, 'Ghost_Ensemble')

    dx          = np.array([50, 50])
    row_well    = 5
    col_well    = 9
    well_loc    = create_wells(row_well, col_well, dx)
    
    q_idx       = [9,13,27,33,41]
    
    mask        = np.full(len(well_loc),True,dtype=bool)
    mask[q_idx] = False
    years = 1
    
    cov_mods = ['Exponential', 'Matern', 'Gaussian']
    computer = ['office', 'binnac']
    setup = computer[0]
    if setup == 'office':
        n_mem  = 40
        nprocs = np.min([n_mem, psutil.cpu_count()]) -1
        inspection = False
        up_temp = True
        printf = True
        spinup = False
        shadow = False
        if n_mem == 2:
            nprocs = 1
            up_temp = False
        
    elif setup == 'binnac':
        n_mem  = 360
        nprocs = psutil.cpu_count()
        up_temp = True
        printf = False
        inspection = False
        spinup = True
        shadow = True
    
    choice_static = 0
    cov_variants = [['cov_data', 'npf'], ['cov_data'], ['npf']]

    valt_variants = ["random_good", "random_low", "random_high", "prior_meas"]
    choice_valt = 0 
    covt_variants = ["good", "random_low", "random_high"]
    choice_covt = 2 
    pp_flag = True 
    pilot_point_even = True
   
    
    if valt_variants[choice_valt] == "prior_meas":
        field_meas_flag = True
    else:
        field_meas_flag = False
    val_first = False
    conditional_flag = True
    scramble_pp = False
    nPP = 45
    l_red = 1
    h_damp = 0.3
    cov_damp = [0.1, 0.1]
    npf_damp = 0.1
    asimdays = [0, 300]
    
    if val_first:
        damp_choice = [[h_damp, npf_damp], [h_damp, cov_damp, npf_damp]]
        est_dat = cov_variants[choice_static].copy()
        est_dat.remove('cov_data')
        cov_choice = [est_dat, cov_variants[choice_static]]
    else:
        damp = [[h_damp, cov_damp, npf_damp], [h_damp, cov_damp], [h_damp, npf_damp]]
        cov_choice = cov_variants[choice_static]
        damp_choice = damp[choice_static]

    if field_meas_flag:
        
        if nPP == 28:
            if pilot_point_even:
                meas_loc = np.array([2, 5, 16, 19, 25])
            else:
                meas_loc = np.array([2, 5, 16, 19, 24])
        elif nPP == 45:
            if pilot_point_even:
                meas_loc = np.array([2, 33, 16, 19, 41])
            else:
                meas_loc = np.array([3, 33, 27, 22, 42])
        
        # np.random.seed(94)
        # meas_loc = np.random.choice(np.arange(0, nPP), np.min([nPP,5]), replace=False)
    else:
        meas_loc = []
    
        
    pars    = {
        'Erdalx': False,
        'refine': False,
        'pilotp': pp_flag,
        'nprocs': nprocs,
        'setup' : setup,
        'shadow': shadow,
        'EnKF_p': cov_choice, 
        'damp'  : damp_choice,
        'val1st': val_first,
        'valday': 15,
        'n_PP'  : nPP,
        'eps'   : 0.01,
        'omitc' : 0,
        'nearPP': 4,
        'sig_me': 0.01,
        'geomea': 1,
        'years' : years,
        'condfl': conditional_flag,
        'covtyp': covt_variants[choice_covt],
        'valtyp': valt_variants[choice_valt],
        'up_tem': up_temp,
        'nx'    : np.array([100, 50]),                      # number of cells
        'dx'    : dx,                                       # cell size
        'wel_k' : False,
        'rch_is': False,                                    # Flag rch isotropy
        'l_red' : l_red,
        'lx'    : np.array([[1100,600], [2500,500]])/l_red, # corellation lengths
        'ang'   : np.array([111, 17]),                      # angle in ° (logK, recharge)
        'sigma' : np.array([1.7, 0.1]),                     # variance (logK, recharge)
        'mu'    : np.array([-8.5, -0.7]),                   # mean (log(ms-1), (mm/d))
        'cov'   : cov_mods[1],                              # Covariance models
        'nlay'  : np.array([1]),                            # Number of layers
        'bot'   : np.array([0]),                            # Bottom of aquifer
        'top'   : np.array([50]),                           # Top of aquifer
        'welxy' : np.array(well_loc[q_idx]),                # location of pumps
        'obsxy' : np.array(well_loc),                       # location of obs
        'welq'  : np.array([9, 18, 90, 0.09, 0.9])/3600,    # Q of wells [m3s-1]
        'welst' : np.array([20, asimdays[1], 200, 0, 0]),   # start day of pump
        'welnd' : np.array([150, 365, 365, 200, 300]),      # end day of pump
        'welay' : np.array(np.zeros(5)),                    # layer of wells
        'river' : np.array([[0.0,0], [5000,0]]),            # start / end of river
        'rivC'  : 1e-3,                                     # river conductance [ms-1]
        'rivd'  : 2,                                        # depth of river [m]
        'chd'   : np.array([[0.0,2500], [5000,2500]]),      # start / end of river
        'chdh'  : 15,                                       # initial stage of riv
        'ss'    : 1e-3,                                     # specific storage
        'sy'    : 0.20,                                     # specific yield
        'asim_d': asimdays,
        'spinup': spinup,
        'mname' : "Reference",
        'sname' : "Reference",
        'f_meas': field_meas_flag,
        'inspec': inspection,
        'printf': printf,
        'ppeven': pilot_point_even,
        'f_m_id': meas_loc,
        'scramb': scramble_pp,
        'sim_ws': os.path.join(Vrdir, 'model_files'),
        'vr_h_d': os.path.join(Vrdir, 'model_data', 'head_ref.npy'),
        'vr_o_d': os.path.join(Vrdir, 'model_data', 'obs_ref.npy'),
        'gg_ws' : os.path.join(Vrdir, 'gridgen_files'),
        'ens_ws': ensemb_dir,
        'mem_ws': os.path.join(ensemb_dir, 'member'),
        'sh_ens': ghost_ensemb_dir,
        'sh_mem': os.path.join(ghost_ensemb_dir, 'member'),
        'timuni': 'SECONDS',                                   # time unit
        'lenuni': 'METERS',                                   # length unit
        'k_r_d' : os.path.join(Vrdir, 'model_data','logK_ref.csv'),
        'r_r_d' : os.path.join(Vrdir, 'model_data','rech_ref.csv'),
        'rh_d'  : os.path.join(Vrdir, 'model_data','tssl.csv'),
        'sf_d'  : os.path.join(Vrdir, 'model_data','sfac.csv'),
        'n_mem' : n_mem,
        'tm_ws' : os.path.join(ensemb_dir, 'template_model'),
        'trs_ws': os.path.join(Vrdir, 'transient_model'),
        'resdir': os.path.join(parent_directory, 'output'),
        'logfil': os.path.join(parent_directory, 'output', 'logs.txt'),
        'nsteps': int(years*365*24/6),
        'rotmat': rotation_matrix,
        'mat2cv': extract_truth,
        'rot2df': rotate2Dfield,
        'covmat': covariance_matrix,
        'dstmat': distance_matrix,
        'period': period,
        'backnd': "threading",
        }
    
    if choice_static == 0 or choice_static == 1:
        if not pp_flag:
            print("You cant have a variogram with no pilotpoints - yet")
            print("Exiting...")
            sys.exit() 
            
    return pars
