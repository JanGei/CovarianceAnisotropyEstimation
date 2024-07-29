import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from plot_angles import plot_angles
from dependencies.randomK import randomK
from dependencies.create_k_fields import create_k_fields
from dependencies.create_pilot_points import create_pilot_points, create_pilot_points_even
from dependencies.compare_conditional import compare_conditional

pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy, near_dist = create_pilot_points_even(gwf, pars)
k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')

lx = np.array([1000,400])
# angles = np.deg2rad(np.arange(0, 180, 10))
angles = [np.deg2rad(17)]

results = []

pars['valt'] = "good"

for i in range(len(angles)):
    res = create_k_fields(gwf, pars, k_ref, pp_xy, pp_cid, test_cov = [lx, angles[i]], conditional = False)
    results.append(res)
    
    # alt_field = create_k_fields(gwf, pars, k_ref, pp_xy, pp_cid, test_cov = [lx, angles[i]], conditional = True)[4]
    # compare_conditional(gwf, pars, res[0], angles[i], res[3][1], pp_xy, np.exp(res[4])/100)
    k_ref_2D = np.flip(np.reshape(k_ref,(50,100)), axis = 0)
    compare_conditional(gwf, pars, res[0], angles[i], res[3][1], pp_xy, k_ref_2D)
