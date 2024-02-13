# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""
import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functions.generator import gsgenerator
import flopy

# imports from parent directory
import sys
sys.path.append('..')
from dependencies.model_params import get
from dependencies.plot import plot_fields

#%% Field generation (based on Olafs Skript)
# Watch out as first entry corresponds to y and not to x

pars = get()
nx      = pars['nx']
dx      = pars['dx']
lx      = pars['lx']
ang     = pars['ang']
sigma   = pars['sigma']
mu      = pars['mu']
cov     = pars['cov']

sim_ws = pars['sim_ws']
mname = pars['mname']


sim        = flopy.mf6.modflow.MFSimulation.load(
                        version             = 'mf6', 
                        exe_name            = 'mf6',
                        sim_ws              = sim_ws, 
                        verbosity_level     = 0
                        )

gwf = sim.get_model(mname)
mg = gwf.modelgrid
xyz = mg.xyzcellcenters
xyzip = list(zip(xyz[0], xyz[1]))


#%% Field generation (based on gstools)

logK    = gsgenerator(gwf, nx, dx, lx[0], ang[0], sigma[0],  cov, random = False) 
logK        = logK.T + mu[0]    # [log(m/s)]
rech    = gsgenerator(gwf, nx, dx, lx[1], ang[1], sigma[1],  cov, random = False) 
rech        = (rech.T + mu[1])  # [mm/d]

# Anmerkung des Übersetzers: Beim generieren dieser Felder ist die Varianz per se dimensionslos
# Wenn wir also die Felder von Erdal und Cirpka nachbilden wollen, müssen wir überhaupt nicht
# die Varianz mitscalieren, wenn die Einheiten geändert werden, sonder nur der mean
inspection = True
if inspection:
    plt.scatter(xyz[0], xyz[1], c=logK)
    plt.show()
    
    plot_fields(gwf, logK, rech)
#%% plotting

print(mu)
print(np.mean(logK), np.mean(rech))
#%% Saving the fields - Übergabe in (m/s)
np.savetxt('model_data/logK_ref.csv', np.exp(logK), delimiter = ',')
np.savetxt('model_data/rech_ref.csv', rech/1000/86400, delimiter = ',')
