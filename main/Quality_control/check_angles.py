import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from plot_angles import plot_angles
from dependencies.randomK import randomK
from dependencies.create_k_fields import create_k_fields
from dependencies.create_pilot_points import create_pilot_points
from dependencies.compare_conditional import compare_conditional

pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy, neardist = create_pilot_points(gwf, pars)

lx = np.array([750,100])
angles = np.deg2rad(np.arange(0, 180, 10))

results = []

pars['valt'] = "good"

for i in range(len(angles)):
    res = create_k_fields(gwf, pars, pp_xy, pp_cid, test_cov = [lx, angles[i]])
    results.append(res)
    # if i == 0:
    #     plot_angles(gwf, pars, res[0], res[1], angles[i], res[3], ref = True)

    # plot_angles(gwf, pars, res[0], res[2], angles[i], res[3])
    alt_field = randomK(angles[i], pars['sigma'][0], pars['cov'], pars['mu'][0], pars, grid = [pars['nx'], pars['dx'], pars['lx'][0]],  ftype = 'K')
    compare_conditional(gwf, pars, res[0], angles[i], res[3][1], pp_xy, np.exp(res[4])/100)
    # compare_conditional(gwf, pars, res[0], angles[i], res[3][1], pp_xy, alt_field/100)
