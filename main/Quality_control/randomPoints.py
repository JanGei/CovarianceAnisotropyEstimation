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
from dependencies.randomK_points import randomK_points
from dependencies.plotting.plot_fields import plot_fields

pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy, near_dist = create_pilot_points(gwf, pars)


l_angs = [5000, 2500, np.deg2rad(20)]
mg = gwf.modelgrid
cxy        = np.vstack((mg.xyzcellcenters[0], mg.xyzcellcenters[1])).T
dxmin      = np.min([max(sublist) - min(sublist) for sublist in mg.xvertices])
dymin      = np.min([max(sublist) - min(sublist) for sublist in mg.yvertices])
dx         = [dxmin, dymin]
lx         = [l_angs[0], l_angs[1]]
ang         = l_angs[2]
sigma   = pars['sigma'][0]

Kflat, K  = randomK_points(mg.extent, cxy, dx,  lx, ang, sigma, pars)

print(np.mean(Kflat))
plot_fields(gwf, pars, np.log(Kflat), Kflat)