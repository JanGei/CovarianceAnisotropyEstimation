import flopy
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_POI(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pp_xy, pars, bc = False):
    
    pad = 0.1
    # welxy   = pars['welxy']
    obsxy   = pars['obsxy']
    kmin    = pars['kmin']
    kmax    = pars['kmax']
  
    fig, axes   = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    
    ax0         = flopy.plot.PlotMapView(model=gwf, ax=axes)
    c           = ax0.plot_array(np.log(gwf.npf.k.array), cmap=cm.bilbao_r, alpha=1)
    axes.scatter(pp_xy[:,0], pp_xy[:,1], marker = '*', color = 'black', label = 'pilot point', s = 20)
    # axes.scatter(welxy[:,0], welxy[:,1], marker = 'o', color = 'blue', label = 'well', s = 50)
    axes.scatter(obsxy[:,0], obsxy[:,1], marker = 'v', color = 'red', label = 'observation', s = 30)
    if bc:
        ax0.plot_bc(name     = 'WEL',
                   package  = gwf.wel,
                   color    = 'blue',
                   label    = 'well')
        ax0.plot_bc(name     = 'RIV',
                   package  = gwf.riv,
                   color    = 'yellow')
        ax0.plot_bc(name     = 'CHD',
                   package  = gwf.chd,
                   color    = 'red')
    axes.legend()
    divider     = make_axes_locatable(axes)
    cax         = divider.append_axes("right", size="3%", pad=pad)
    cbar        = fig.colorbar(c, cax=cax)
    cbar.mappable.set_clim(kmin, kmax)
    
    cbar.set_label('Log-Conductivity (log(m/s))')
    axes.set_aspect('equal')
    axes.set_ylabel('Y-axis')