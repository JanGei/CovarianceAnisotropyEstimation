import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from plot_angles import plot_angles
# from dependencies.randomK_points import randomK_points
from dependencies.conditional_k import conditional_k
from dependencies.create_pilot_points import create_pilot_points


pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy = create_pilot_points(gwf, pars)

lx = np.array([700,150])
angles = np.deg2rad(np.arange(20, 160, 10))

results = []

pars['valt'] = "good"

for i in range(len(angles)):
    res = conditional_k(gwf, pars, pp_xy, pp_cid, test_cov = [lx, angles[i]])
    results.append(res)
    plot_angles(gwf, pars, res[0], res[1], angles[i], res[3])
