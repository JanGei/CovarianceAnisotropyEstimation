import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
import flopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize

def compare_conditional(gwf, pars, logk_proposal, angle, condp, ppxy, K):
    
    kmin    = np.min(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    kmax    = np.max(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    pad = 0.1
    
    # Create meshgrid
    X, Z = np.meshgrid(np.arange(0, (pars['nx'][0]+1)*pars['dx'][0], pars['dx'][0]), 
                       np.arange(0, (pars['nx'][1]+1)*pars['dx'][1], pars['dx'][1]))

    # Plot the data - CHECK THIS!!
    plotK = np.hstack((K, K[:,-1].reshape(len(K[:,-1]),1)))
    k_add = np.hstack((K[-1,:], K[-1,-1]))
    plotK = np.vstack((plotK, k_add))   
     
    fig, axes   = plt.subplots(nrows=2, ncols=1, figsize=(8,6))

    ax0         = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
    c           = ax0.plot_array(np.log(logk_proposal), cmap=cm.bilbao_r, alpha=1)
    divider     = make_axes_locatable(axes[0])
    cax         = divider.append_axes("right", size="3%", pad=pad)
    cbar0       = fig.colorbar(c, cax=cax)
    cbar0.mappable.set_clim(kmin, kmax)
    cbar0.set_label('Log-Conductivity (log(m/s))')
    
    norm = Normalize(vmin=kmin, vmax=kmax)
    axes[0].scatter(ppxy[:,0], ppxy[:,1], c=condp, cmap=cm.bilbao_r, norm=norm)
    axes[0].set_aspect('equal')
    axes[0].set_ylabel('Y-axis')
    axes[0].set_title(f'Angle: {np.rad2deg(angle)}')
    
    pl = axes[1].pcolor(X, Z, np.log(plotK), cmap = cm.bilbao_r, vmin = kmin, vmax = kmax)
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    plt.colorbar(pl)
    
    norm = Normalize(vmin=kmin, vmax=kmax)
    axes[1].scatter(ppxy[:, 0], ppxy[:, 1], c=condp, cmap=cm.bilbao_r, s=20, edgecolors='black', linewidths=1.5, norm=norm)
    axes[0].scatter(ppxy[:, 0], ppxy[:, 1], c=condp, cmap=cm.bilbao_r, s=20, edgecolors='black', linewidths=1.5, norm=norm)

    
    plt.show()
        
    
    
