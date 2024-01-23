# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""

import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt
from functions.generator import gsgenerator
from functions.model_params import get
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


#%% Field generation (based on gstools)

X,Y,logK    = gsgenerator(nx, dx, lx[0], ang[0], sigma[0],  cov, random = False) 
logK        = logK.T + mu[0]    # [log(m/s)]
X,Y,rech    = gsgenerator(nx, dx, lx[1], ang[1], sigma[1],  cov, random = False) 
rech        = (rech.T + mu[1])  # [mm/d]

# Anmerkung des Übersetzers: Beim generieren dieser Felder ist die Varianz per se dimensionslos
# Wenn wir also die Felder von Erdal und Cirpka nachbilden wollen, müssen wir überhaupt nicht
# die Varianz mitscalieren, wenn die Einheiten geändert werden, sonder nur der mean

#%% plotting
cmaps = ['Blues', 'BuPu', 'CMRmap', 'Grays', 'OrRd', 'RdGy', 'YlOrBr', 'afmhot',
        'cividis', 'copper']

cmapc = ['batlowK', 'bilbao', 'berlin', 'devon', 'glasgow', 'grayC', 'lajolla',
         'lapaz', 'lipari', 'nuuk', 'oslo', 'turku']

cmnam = cm.cmaps
names = list(cmnam.keys())

windowx = np.array([0, 5000])
windowy = np.array([0, 2500])
mask_x = (X >= windowx[0]) & (X <= windowx[1])
mask_y = (Y >= windowy[0]) & (Y <= windowy[1])
mask_combined = np.ix_(mask_y[:, 0], mask_x[0, :])

cmap_rech = cm.turku_r
cmap_logK = cm.bilbao_r

pad = 0.1
d = 3600

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=pad)
cbar = fig.colorbar(ax1.pcolor(X, Y, logK, cmap=cmap_logK), cax=cax)
cbar.set_label('Log-Conductivity (log(m/s))')
ax1.set_ylabel('Y-axis')
ax1.set_aspect('equal')

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="3%", pad=pad)  
cbar = fig.colorbar(ax2.pcolor(X, Y, rech, cmap=cmap_rech), cax=cax)
cbar.set_label('Recharge (m/d)')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_aspect('equal')

print(mu)
print(np.mean(logK), np.mean(rech))
#%% Saving the fields - Übergabe in (m/s)
np.savetxt('model_data/logK_reference.csv', np.exp(logK), delimiter = ',')
np.savetxt('model_data/rech_reference.csv', rech/1000/86400, delimiter = ',')
