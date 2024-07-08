import sys
sys.path.append('..')
import numpy as np
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.randomK import randomK
from dependencies.create_k_fields import create_k_fields
from dependencies.create_pilot_points import create_pilot_points
from dependencies.compare_conditional import compare_conditional
from dependencies.plotting.plot_k_fields import plot_k_fields

pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy, neardist = create_pilot_points(gwf, pars)

lx = np.linspace(500,1300,200)
angle = np.deg2rad(pars['ang'][0])

results = []
k_fields = []
pars['valt'] = "good"

for i in range(10):
    res = create_k_fields(gwf, pars, pp_xy, pp_cid, test_cov = [[lx[10],100], angle])
    results.append(res)
    k_fields.append(res[0])

    
    alt_field = randomK(angle, pars['sigma'][0], pars['cov'], pars['mu'][0], pars, grid = [pars['nx'], pars['dx'], np.array([lx[i],100])],  ftype = 'K')
    compare_conditional(gwf, pars, res[0], angle, np.exp(res[3][1]), pp_xy, np.exp(res[4])/100)

# plot_k_fields(gwf, pars,  k_fields)
