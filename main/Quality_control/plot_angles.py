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
 
def plot_angles(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pars,  logk_proposal: np.ndarray, matv, angle, condp):
    
    pars = get()
    kmin    = np.min(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    kmax    = np.max(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    
    pad = 0.1
    
    mat = np.zeros((2,2))
    mat[0,0] = matv[0]
    mat[0,1] = matv[1]
    mat[1,0] = matv[1]
    mat[1,1] = matv[2]
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    # lx1, lx2, ang = extract_truth(eigenvalues, eigenvectors)
    lx1, lx2, ang = pars['mat2cv'](eigenvalues, eigenvectors)
    l = max(lx1, lx2) * 1.5
    center = (0,0)
  
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
    axes[0].set_title(f'Angle: {np.rad2deg(angle)} and {np.rad2deg(ang)}')
    
    ellipse = patches.Ellipse(center,
                              lx1*2,
                              lx2*2,
                              angle=np.rad2deg(ang),
                              fill=False,
                              color='black',
                              alpha = 0.5,
                              zorder = 1)
    axes[1].add_patch(ellipse)
    axes[1].set_xlim(-l, l)
    axes[1].set_ylim(-l, l)
    axes[1].set_aspect('equal', 'box')
    
    plt.show()