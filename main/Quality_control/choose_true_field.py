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
from dependencies.plotting.plot_k_fields import plot_k_fields
from dependencies.compare_conditional import compare_conditional
from scipy.interpolate import griddata


random_seeds = np.arange(0,100,1)

pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy, neardist = create_pilot_points(gwf, pars)

mg = gwf.modelgrid
cxy = np.vstack((mg.xyzcellcenters[0], mg.xyzcellcenters[1])).T

n = 38
x = np.arange(pars['dx'][0]/2, pars['nx'][0]*pars['dx'][0], pars['dx'][0])
y = np.arange(pars['dx'][1]/2, pars['nx'][1]*pars['dx'][1], pars['dx'][1])

# Grid in Physical Coordinates
X, Y = np.meshgrid(x, y)

ang = np.deg2rad(pars['ang'][0])

for i in range(n):

    

    # plot_angles(gwf, pars, res[0], res[2], angles[i], res[3])
    K1 = randomK(ang, pars['sigma'][0], pars['cov'], pars['mu'][0],
                        pars, grid = [pars['nx'], pars['dx'], pars['lx'][0]],  ftype = 'K', randn = i, random = False)

    
    Kflat1 =  griddata((X.ravel(order = 'F'), Y.ravel(order = 'F')), K1.ravel(order = 'F'),
                     (cxy[:,0], cxy[:,1]), method='nearest')

    
    # print(pars['mu'][0], np.mean(np.log(K1)))
    # print(pars['mu'][1]/(86400), np.mean(K2))
    plot_k_fields(gwf, pars, [Kflat1, Kflat1])

