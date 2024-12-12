import sys
sys.path.append('..')
import numpy as np
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.randomK import randomK
from dependencies.create_k_fields import create_k_fields
from dependencies.create_pilot_points import create_pilot_points, create_pilot_points_even
from dependencies.compare_conditional import compare_conditional
from dependencies.plotting.plot_k_fields import plot_k_fields
from dependencies.conditional_k import conditional_k
pars = get()
sim, gwf = load_template_model(pars)
if pars['ppeven']:
    pp_cid, pp_xy = create_pilot_points_even(gwf, pars)
else:
    pp_cid, pp_xy = create_pilot_points(gwf, pars)

k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')
lx = np.linspace(500,1300,10)
angle = np.deg2rad(pars['ang'][0])
clx = pars['lx']          # reference correlation length
angles = pars['ang']      # reference anisotropy angle
sigma = pars['sigma'][0]  # reference variance
mu = pars['mu'][0]        # reference k mean
dx = pars['dx']           # model discretization
mg = gwf.modelgrid        # modelgrid
xyz = mg.xyzcellcenters   # xyz coordinates of cells
cxy = np.vstack((xyz[0], xyz[1])).T # reshaping (x,y) coordinates
sig_meas = pars['sig_me'] # measurement uncertainty
mean_range = np.log(3)    # range from which to draw the mean
results = []
k_fields = []
pars['valt'] = "good"

# for j in range(6):
#     results.append(create_k_fields(gwf, pars, k_ref,pp_xy, pp_cid, test_cov = [[1100,600], angle]))
#     k_fields.append(results[j][0])
#     print(round(np.mean(np.log(k_fields[j])),2), round(np.var(np.log(k_fields[j])),2))
#     print('_______')
# plot_k_fields(gwf, pars,  k_fields)

for j in range(5):
    res = create_k_fields(gwf, pars, k_ref,pp_xy, pp_cid)
    pp_k = np.log(res[0][pp_cid])
    sigma = np.var(pp_k)
    print(sigma)
    cond_field = conditional_k(cxy, dx, [1100,600], angle, sigma, pars, pp_k, pp_xy)
    compare_conditional(gwf, pars, res[0], angle, np.log(res[3][1]), pp_xy, cond_field[0])
    
    


