import sys
sys.path.append('..')
sys.path.append('../main')

import numpy as np
import flopy
from cmcrameri import cm
import matplotlib.patches as patches
from dependencies.randomK import randomK
from dependencies.create_pilot_points import create_pilot_points, create_pilot_points_even
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from functions.kriging import kriging
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from Virtual_Reality.ReferenceModel import create_reference_model


pars = get()
create_reference_model(pars)
sim, gwf = load_template_model(pars, SS = True)
if pars['ppeven']:
    pp_cid, pp_xy = create_pilot_points_even(gwf, pars)
else:
    pp_cid, pp_xy = create_pilot_points(gwf, pars)
dx = pars['dx']
sigma = pars['sigma'][0]
cxy = np.vstack([gwf.modelgrid.xcellcenters, gwf.modelgrid.ycellcenters]).T

x = np.arange(pars['dx'][0]/2, pars['nx'][0]*pars['dx'][0], pars['dx'][0])
y = np.arange(pars['dx'][1]/2, pars['nx'][1]*pars['dx'][1], pars['dx'][1])

# Grid in Physical Coordinates
X, Y = np.meshgrid(x, y)

K = randomK(np.deg2rad(pars['ang'][0]), pars['sigma'][0], pars['cov'], pars, ftype = 'K', random = False)
k_true =  griddata((X.ravel(order = 'F'), Y.ravel(order = 'F')), K.ravel(order = 'F'),
                 (cxy[:,0], cxy[:,1]), method='nearest')


pp_k = [np.log(k_true[pp_cid]),
        np.log(k_true[pp_cid])* np.random.uniform(0.9,1.1,k_true[pp_cid].shape),
        np.random.normal(pars['mu'][0], np.sqrt(pars['sigma'][0]), len(pp_cid))]
print(pp_k[1][[5]])
lx = [pars['lx'][0], np.array([1600,400])]
ang = [np.deg2rad(pars['ang'][0]), np.deg2rad(3)]

v_a_c1, f_g1 = kriging(cxy, dx, lx[0], ang[0], sigma, pars, pp_k[0], pp_xy)
v_a_c2, f_g2 = kriging(cxy, dx, lx[1], ang[1], sigma, pars, pp_k[1], pp_xy)
v_a_c3, f_g3 = kriging(cxy, dx, lx[1], ang[1], sigma, pars, pp_k[2], pp_xy)

fields = [k_true, v_a_c1, v_a_c2, v_a_c3]
lengths = [lx[0], lx[0], lx[1], lx[1]]
angles = [ang[0], ang[0], ang[1], ang[1]]

hfields = []
for i, field in enumerate(fields):
    gwf.npf.k.set_data([field])
    gwf.sto.transient.set_data({0: False})
    gwf.sto.steady_state.set_data({0: True})
    sim.write_simulation()
    sim.run_simulation()
    hfields.append(gwf.output.head().get_data())
    if i == -1:
        gwf.npf.k.set_data([k_true])
        gwf.npf.write()

#%% plot        
nrows, ncols = 3,4
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex = True, sharey = True, figsize=(16.5,6.5), constrained_layout=True)
ax0 = flopy.plot.PlotMapView(model=gwf, ax=axes[0,0])
c0 = ax0.plot_array(np.log(k_true), cmap=cm.bilbao_r, alpha=1)


for i in range(ncols):

    ax1 = flopy.plot.PlotMapView(model=gwf, ax=axes[0,i])
    c1 = ax1.plot_array(np.log(fields[i]), cmap=cm.bilbao_r, alpha=1)
    axes[0,i].set_aspect('equal')
    axes[0,i].scatter(pp_xy[:,0],pp_xy[:,1], color = 'black', s = 5)
    
    ax = flopy.plot.PlotMapView(model=gwf, ax=axes[1,i])
    c2 = ax.plot_array((hfields[i]), cmap=cm.devon_r, alpha=1)
    axes[1,i].set_aspect('equal')
    # axes[0,0].scatter(pp_xy[:,0], pp_xy[:,1], c='black', marker='x', s=10)
            
    ellipse = patches.Ellipse((2500,1250),
                              lengths[i][0]*2,
                              lengths[i][1]*2,
                              angle=np.rad2deg(angles[i]),
                              fill=False,
                              color='black',
                              alpha = 0.5,
                              zorder = 1)
    axes[2,i].add_patch(ellipse)
    axes[2,i].set_aspect('equal')
    
for j in range(nrows):
    # axes[0, j].sharex(axes[1, j])
    # axes[0, j].sharey(axes[1, j])
    for i in range(ncols):
        axes[j,i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[j,i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.001, wspace=0.3)        
        
# divider0 = make_axes_locatable(axes[0])
# cax0 = divider0.append_axes("right", size="5%", pad=pad)  # Adjust size and pad for better spacing
# cbar0 = fig.colorbar(c0, cax=cax0)
# cbar0.mappable.set_clim(kmin, kmax)
# cbar0.set_label('Log-Conductivity (log(m/s))', fontsize=fontsize)
# cbar0.ax.tick_params(labelsize=fontsize)
# axes[0].set_aspect('equal')  # Change to 'auto' to prevent squishing
# axes[0].set_ylabel('Y-axis', fontsize=fontsize)
# axes[0].tick_params(axis='both', which='major', labelsize=fontsize)