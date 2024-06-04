'''
This script checks, how efficient we can draw functional covaraince models from
either a parametric or a normal covariance representation and whether we end up
with non-physical results.
'''

import sys
sys.path.append('..')
import numpy as np
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.create_pilot_points import create_pilot_points
from joblib import Parallel, delayed

def check_new_matrix(data):
    mat = np.zeros((2,2))
    mat[0,0] = data[0]
    mat[0,1] = data[1]
    mat[1,0] = data[1]
    mat[1,1] = data[2]
    
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    
    #check for positive definiteness
    if np.all(eigenvalues > 0):
        pos_def = True
    else:
        pos_def = False
        
    return eigenvalues, eigenvectors, mat, pos_def


def get_cov_mod(gwf, pars: dict, pp_xy = [], pp_cid = [], covtype = 'random', valtype = 'good', test_cov = []):
    clx     = pars['lx']
    angles  = pars['ang']
    mu      = pars['mu'][0]
    k_ref   = np.loadtxt(pars['k_r_d'], delimiter = ',')
    
    sig_meas = 0.1 # standard deviation of measurement error
    
    if covtype == 'random':
        lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]*2),
                       np.random.randint(pars['dx'][1], clx[0][1]*2)])
        ang = np.random.uniform(0, np.pi)
        if lx[0] < lx[1]:
            lx = np.flip(lx)
            if ang > np.pi/2:
                ang -= np.pi/2
            else:
                ang += np.pi/2
        elif lx[0] == lx[1]:
            lx[0] += 1
    elif covtype == 'good':
        lx = clx[0]
        ang = np.deg2rad(angles[0])
    elif covtype == 'test':
        lx = test_cov[0]
        ang = test_cov[1]
        
    # starting k values at pilot points
    if valtype == 'good':
        pp_k = np.log(k_ref[pp_cid.astype(int)])
        # Kg = k_ref_m
        
    elif valtype == 'random':
        pp_k = np.random.uniform(mu-mu/4, mu+mu/4, len(pp_cid))
        pp_k = pp_k + sig_meas * np.random.randn(*pp_k.shape)

    D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
    return [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang]


pars        = get()
n_mem       = pars['n_mem']
nprocs      = pars['nprocs']

sim, gwf = load_template_model(pars)

cor_ellips = []
l_angs = []

if pars['pilotp']:
    pp_cid, pp_xy = create_pilot_points(gwf, pars)
    # create_k_fields
    result = Parallel(n_jobs=nprocs, backend = "threading")(delayed(get_cov_mod)(
        gwf,
        pars, 
        pp_xy,
        pp_cid,
        covtype = pars['covt'],
        valtype = pars['valt']
        )
        for idx in range(n_mem)
        )
    # sorting the results
    for tup in result:
        ellips, l_ang = tup
        cor_ellips.append(ellips)
        l_angs.append(l_ang)

mean_cov = np.mean(np.array(l_angs), axis = 0)
var_cov = np.var(np.array(l_angs), axis = 0)
mean_cov_par = np.mean(np.array(cor_ellips), axis = 0)
var_cov_par = np.var(np.array(cor_ellips), axis = 0)


target_n = 100
result = []
# compare_results = np.zeros(target_n, 6)
n_decimals = 2
print(f'Mean statistics: {mean_cov[0]:.{n_decimals}f} m, {mean_cov[1]:.{n_decimals}f} m, {np.rad2deg(mean_cov[2]):.{n_decimals}f}° ')
for i in range(target_n):
    pos_def = False
    counter = 0    
    while not pos_def:
        a = np.random.normal(mean_cov_par[0], np.sqrt(var_cov_par[0]))
        b = np.random.normal(mean_cov_par[1], np.sqrt(var_cov_par[1]))
        m = np.random.normal(mean_cov_par[2], np.sqrt(var_cov_par[2]))
        eigenvalues, eigenvectors, mat, pos_def = check_new_matrix([a,b,m])
        counter += 1
        
    result.append(counter)
    l1, l2, ang = pars['mat2cv'](eigenvalues, eigenvectors)
    print(f'Obtained {l1:.{n_decimals}f} m, {l2:.{n_decimals}f} m and {np.rad2deg(ang):.{n_decimals}f}° after {counter} tries')

print(f'It took {np.mean(result)} tries to find a suitable replacemment on average')      


