import sys
sys.path.append('..')
sys.path.append('../main')

import numpy as np
import flopy
from cmcrameri import cm
import os
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dependencies.create_pilot_points import create_pilot_points, create_pilot_points_even
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
import matplotlib.pyplot as plt
from Virtual_Reality.ReferenceModel import create_reference_model

cwd = os.getcwd()
pars = get()
create_reference_model(pars)
sim, gwf = load_template_model(pars, SS = True)
if pars['ppeven']:
    pp_cid, pp_xy = create_pilot_points_even(gwf, pars)
else:
    pp_cid, pp_xy = create_pilot_points(gwf, pars)
    
k_ref = np.genfromtxt
river_stages    = np.genfromtxt(pars['rh_d'],delimiter = ',', names=True)['Wert']
k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')
r_ref = np.loadtxt(pars['r_r_d'], delimiter = ',')
sfac  = np.genfromtxt(pars['sf_d'],delimiter = ',', names=True)['Wert']
obsxy = pars['obsxy'] + np.array([-25, 25])
welxy = pars['welxy'] + np.array([-25, 25])
# %% plot
yearin6hrs = 365*4
fontsize = 12
m_s_to_mm_day = 1000*86400
xticks = [1000, 2500, 4000]
yticks = [1000, 2000]

xticks_days = [100*4, 200*4, 300*4]
xlabels_days = [100, 200, 300]

pad = 0.2
fig = plt.figure(figsize=(14,8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])

# Upper left plot
ax1 = fig.add_subplot(gs[0, 0])
fp1 = flopy.plot.PlotMapView(model=gwf, ax=ax1)
map_view = flopy.plot.PlotMapView(model=gwf)
map_view.plot_grid(ax=ax1, alpha = 0.5, zorder = 1)
ax1.scatter(obsxy[:,0],obsxy[:,1], c = 'black', marker = 's', s  =10, zorder = 2)
# fp1.plot_bc(name    = 'WEL',
#            package  = gwf.wel,
#            color    = 'red',
#            label    = 'well',
#            zorder   = 3)
fp1.plot_bc(name     = 'RIV',
           package  = gwf.riv,
           color    = 'blue')
fp1.plot_bc(name     = 'CHD',
           package  = gwf.chd,
           color    = 'green')
for i in range(len(welxy)):
    ax1.scatter(welxy[:,0],welxy[:,1], c = 'red', marker = 's', s  = 10, zorder = 3)
    hollow_circle = patches.Circle(welxy[i], 100, edgecolor='red', facecolor='none', linewidth=4, zorder = 2)
    ax1.add_patch(hollow_circle)
    ax1.text(welxy[i,0]+75, welxy[i,1]+75, f'{i+1}', fontsize=fontsize+2, fontweight='bold', color='red')

custom_lines = [Line2D([0], [0], color='green', lw=6, markersize=10, label='Constant head'),
                Line2D([0], [0], color='blue', lw=6, markersize=10, label='River'),
                Line2D([0], [0], color='black', lw=0, marker='s', markersize=10, label='Observation well'),
                Line2D([0], [0], color='red', lw=0, marker='s', markersize=10, label='Production well')]
ax1.legend(handles=custom_lines, loc = 'upper right', prop={'size': fontsize -2})    

# Upper right plot [0,1]
ax2 = fig.add_subplot(gs[0, 1])
fp2 = flopy.plot.PlotMapView(model=gwf, ax=ax2)
k = fp2.plot_array(np.log10(gwf.npf.k.array), cmap=cm.bilbao_r, alpha=1)
divider0 = make_axes_locatable(ax2)
cax0 = divider0.append_axes("right", size="5%", pad=pad)  # Adjust size and pad for better spacing
cbar0 = fig.colorbar(k, cax=cax0)
# cbar0.mappable.set_clim(kmin, kmax)
cbar0.set_label(r'Conductivity ($\mathrm{log_{10}\frac{m}{s}}$)', fontsize=fontsize)
cbar0.ax.tick_params(labelsize=fontsize-4)

# Lower left plot [1,0]
ax3 = fig.add_subplot(gs[1, 0])  
ax3.plot(river_stages[:yearin6hrs], c = 'blue')
ax3.set_yticks([13,15,17, 19])
ax3.set_yticklabels([13,15,17, 19], fontsize=fontsize -2, color='blue') 
ax3.set_ylabel('River stage (m)', fontsize=fontsize, color='blue') 

ax3_2 = ax3.twinx()
ax3_2.plot(sfac[:yearin6hrs], c = 'red')
ax3_2.set_yticks([0, 1, 2])
ax3_2.set_yticklabels([0, 1, 2], fontsize=fontsize -2, color='red') 
ax3_2.set_ylabel('Seasonal trend (-)', fontsize=fontsize, color='red') 


# Lower right plot [1,1]
ax4 = fig.add_subplot(gs[1, 1])  
fp4 = flopy.plot.PlotMapView(model=gwf, ax=ax4)
r = fp4.plot_array(r_ref*(-m_s_to_mm_day), cmap=cm.turku_r, alpha=1)
divider1 = make_axes_locatable(ax4)
cax1 = divider1.append_axes("right", size="5%", pad=pad)  # Adjust size and pad for better spacing
cbar1 = fig.colorbar(r, cax=cax1)
cbar1.set_label(r'Recharge ($\mathrm{\frac{m}{s}}$)', fontsize=fontsize)
# cbar1.set_ticks(np.arange(0.4, 1.4, 0.2)* (-1e-8))
# tick_labels = [f'{tick:.2}' for tick in np.arange(0.4, 1.4, 0.2)]
# cbar1.set_ticklabels(tick_labels) 
# cbar1.ax.text(0.5, 1.05, 'x 10⁻⁸', ha='center', va='bottom', fontsize=12, transform=cbar1.ax.transAxes)
cbar1.ax.tick_params(labelsize=fontsize -2)

    
axes = np.array([[ax1, ax2], [ax3, ax4]])

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_aspect('auto')
    
for i in range(2):
    for j in range(2):
        if i == 1 and j == 0:
            axes[i,j].set_xticks(xticks_days)
            axes[i,j].set_xticklabels(xlabels_days, fontsize=fontsize -2)
            axes[i,j].set_xlim([0, yearin6hrs])
            axes[i,j].set_xlabel('Time (day)', fontsize=fontsize)
              

        else:
            axes[i,j].set_ylabel('Northing (m)', fontsize=fontsize)
            axes[i,j].set_xlabel('Easting (m)', fontsize=fontsize)
            axes[i,j].set_xticks(xticks)
            axes[i,j].set_yticks(yticks)
            axes[i,j].set_xticklabels(xticks, fontsize=fontsize -2)
            axes[i,j].set_yticklabels(yticks, fontsize=fontsize -2) 
plt.subplots_adjust(hspace=0.25, wspace=0.25) 

fig.savefig(os.path.join(cwd, 'plots', 'Model_Overview.png'), transparent=True, dpi=300)
