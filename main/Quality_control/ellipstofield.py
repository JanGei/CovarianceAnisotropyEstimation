import sys
sys.path.append('..')
import numpy as np
import matplotlib.patches as patches
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.create_k_fields import create_k_fields
from dependencies.create_pilot_points import create_pilot_points
import matplotlib.pyplot as plt
import flopy
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy, neardist = create_pilot_points(gwf, pars)

# define 3 ellipses
data = np.zeros((3,3))
data[:,0] = np.array([300, 400, 580])
data[:,1] = np.array([150, 200, 105])
data[:,2] = np.array([1, 3, 2])

l = 1.15 * np.max(np.max(data))
results = []

k_ref = np.loadtxt(pars['k_r_d'])

pars['valt'] = 'good'
for i in range(len(data)):
    res = create_k_fields(gwf, pars, k_ref, pp_xy, pp_cid, test_cov = [data[i,0:2], data[i,2]])
    results.append(res)


fs = 20
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10), sharey='row')
plt.subplots_adjust(wspace=0.09, hspace=0.05)
for i in range(3):
    axes[0][i].set_xlabel('Correlation Length 1', fontsize=fs-2)
    axes[0][i].set_xlim([-l, l])
    axes[0][i].set_ylim([-l, l])
    ellipse = patches.Ellipse((0, 0), data[i,0]*2, data[i,1]*2, angle=np.rad2deg(data[i,2]),linewidth=2, fill=False, color='black')
    axes[0][i].add_patch(ellipse)
    axes[0][i].set_aspect('equal')
    axes[0][i].tick_params(axis='x', labelsize=fs-4)
    axes[0][i].tick_params(axis='y', labelsize=fs-4)
    
    axobj  = flopy.plot.PlotMapView(model=gwf, ax=axes[1][i])
    c  = axobj.plot_array(np.log(results[i][0]), cmap=cm.bilbao_r, alpha=1)
    axes[1][i].set_aspect('equal')
    axes[1][i].tick_params(axis='x', labelsize=fs-4)
    axes[1][i].tick_params(axis='y', labelsize=fs-4)
    axes[1][i].set_xlabel('Easting [m]', fontsize=fs-4)
    
axes[0][0].set_ylabel('Correlation Length 2', fontsize=fs-2)    
axes[1][0].set_ylabel('Northing [m]', fontsize=fs-4)


   
kmin  = np.min(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
kmax  = np.max(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))

cbar0 = fig.colorbar(c, ax=[axes[1][0], axes[1][1], axes[1][2]], fraction=0.008, pad=0.01) 
cbar0.mappable.set_clim(kmin, kmax)
cbar0.set_label('Log(K)', fontsize = fs-2)
# Set custom bounds for colorbars
cbar0.mappable.set_clim(vmin=kmin, vmax=kmax)
cbar0.ax.tick_params(labelsize=fs-4)
       




