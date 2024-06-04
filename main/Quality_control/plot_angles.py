import flopy
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import sys
sys.path.append('..')
from dependencies.model_params import get
from matplotlib.colors import Normalize
 
def plot_angles(gwf, pars,  logk_proposal, ellips, angle, condp, ref = False):
    
    pars = get()
    kmin    = np.min(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    kmax    = np.max(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    pad = 0.1
    
    l = max(ellips[0], ellips[1]) * 1.5
    center = (0,0)
    
    if ref == True:
        fig, axes   = plt.subplots(nrows=2, ncols=1)

        ax0         = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
        c           = ax0.plot_array(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')), cmap=cm.bilbao_r, alpha=1)
        divider     = make_axes_locatable(axes[0])
        cax         = divider.append_axes("right", size="3%", pad=pad)
        cbar0       = fig.colorbar(c, cax=cax)
        cbar0.mappable.set_clim(kmin, kmax)
        cbar0.set_label('Log-Conductivity (log(m/s))')
        
        norm = Normalize(vmin=kmin, vmax=kmax)
        axes[0].scatter(condp[0][:,0], condp[0][:,1], c=condp[1], cmap=cm.bilbao_r, norm=norm)
        axes[0].set_aspect('equal')
        axes[0].set_ylabel('Y-axis')
        axes[0].set_title(f'Angle: {pars["ang"][0]} and {pars["ang"][0]}')
        
        ellipse = patches.Ellipse(center,
                                  pars['lx'][0][0]*2,
                                  pars['lx'][0][1]*2,
                                  angle=pars['ang'][0],
                                  fill=False,
                                  color='black',
                                  alpha = 0.5,
                                  zorder = 1)
        axes[1].add_patch(ellipse)
        axes[1].set_xlim(-l, l)
        axes[1].set_ylim(-l, l)
        axes[1].set_aspect('equal', 'box')
        
        plt.show()
        
    fig, axes   = plt.subplots(nrows=2, ncols=1, figsize=(8,6))

    ax0         = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
    c           = ax0.plot_array(np.log(logk_proposal), cmap=cm.bilbao_r, alpha=1)
    divider     = make_axes_locatable(axes[0])
    cax         = divider.append_axes("right", size="3%", pad=pad)
    cbar0       = fig.colorbar(c, cax=cax)
    cbar0.mappable.set_clim(kmin, kmax)
    cbar0.set_label('Log-Conductivity (log(m/s))')
    
    norm = Normalize(vmin=kmin, vmax=kmax)
    axes[0].scatter(condp[0][:,0], condp[0][:,1], c=condp[1], cmap=cm.bilbao_r, norm=norm)
    axes[0].set_aspect('equal')
    axes[0].set_ylabel('Y-axis')
    axes[0].set_title(f'Angle: {np.rad2deg(angle)}')
    
    ellipse = patches.Ellipse(center,
                              ellips[0]*2,
                              ellips[1]*2,
                              angle=np.rad2deg(angle),
                              fill=False,
                              color='black',
                              alpha = 0.5,
                              zorder = 1)
    axes[1].add_patch(ellipse)
    axes[1].set_xlim(-l, l)
    axes[1].set_ylim(-l, l)
    axes[1].set_aspect('equal', 'box')
    
    plt.show()
    