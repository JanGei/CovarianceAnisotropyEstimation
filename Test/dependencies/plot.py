#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:54:48 2024

@author: janek
"""
import matplotlib.pyplot as plt
from cmcrameri import cm
import flopy
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter


def plot(gwf, pkgs, bc = False):
    plotables = {}
    if 'rch' in pkgs:
        rch = {
            'cmap'      : cm.turku_r,
            'data'      : abs(gwf.rch.stress_period_data.get_data()[0]['recharge']),  
            'cbttl'     : 'Recharge (m/s)',                  
            }
        plotables.update({'rch':rch})
    if 'logK' in pkgs:
        logK = {
            'cmap'      : cm.bilbao_r,
            'data'      : np.log(gwf.npf.k.array), 
            'cbttl'     : 'Log-Conductivity (log(m/s))',  
            }
        plotables.update({'logK':logK})
    if 'h' in pkgs:
        h = {
            'cmap'      : cm.devon_r,
            'data'      : gwf.output.head().get_data(), 
            'cbttl'     : 'Hydraulic Head (m)', 
            }
        plotables.update({'h':h})

    pad         = 0.1
    
    fig, axes   = plt.subplots(nrows=len(pkgs), ncols=1, figsize=(4*len(pkgs),6), sharex=True)
    for i, pkg in enumerate(pkgs):
        ax          = flopy.plot.PlotMapView(model=gwf, ax=axes[i])
        c           = ax.plot_array(plotables[pkg]['data'], cmap=plotables[pkg]['cmap'], alpha=1)
        divider     = make_axes_locatable(axes[i])
        cax         = divider.append_axes("right", size="3%", pad=pad)
        cbar        = fig.colorbar(c, cax=cax)
        cbar.set_label(plotables[pkg]['cbttl'])
        axes[i].set_aspect('equal')
        axes[i].set_ylabel('Y-axis')
        if pkg == 'h' and bc == True:
            ax.plot_bc(name     = 'WEL',
                       package  = gwf.wel,
                       color    = 'black')
            ax.plot_bc(name     = 'RIV',
                       package  = gwf.riv,
                       color    = 'yellow')
            ax.plot_bc(name     = 'CHD',
                       package  = gwf.chd,
                       color    = 'red')
        if i == len(pkgs)-1:
            axes[i].set_xlabel('X-axis')
            
def plot_POI(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pp_xy, pars, bc = False):
    
    pad = 0.1
    welxy   = pars['welxy']
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
    
def plot_k_fields(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pars,  k_fields: list):
    
    assert len(k_fields)%2 == 0, "You should provide an even number of fields"
    
    pad = 0.1
    
    layout = [[f'l{i}', f'r{i}'] for i in range(int(len(k_fields)/2))]
    low_plot = ['b', 'b']
    layout.append(low_plot)
    layout.append(low_plot)
    
    fig, axes = plt.subplot_mosaic(layout, figsize=(4,len(k_fields)/2+2), sharex=True, sharey = True)    
    for i in range(int(len(k_fields)/2)):
        for j, letter in enumerate(['r', 'l']):
            gwf.npf.k.set_data(k_fields[i*2+j])
            ax = axes[letter+str(i)]
            axf = flopy.plot.PlotMapView(model=gwf, ax=ax)
            c = axf.plot_array(np.log(gwf.npf.k.array), cmap=cm.bilbao_r, alpha=1)
            ax.set_aspect('equal')
    
    gwf.npf.k.set_data(np.mean(k_fields, axis=0))   
    ax = axes['b']
    axf = flopy.plot.PlotMapView(model=gwf, ax=ax)
    c = axf.plot_array(np.log(gwf.npf.k.array), cmap=cm.bilbao_r, alpha=1)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    

def plot_fields(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pars,  logk_proposal, rech_proposal: np.ndarray):
    
    kmin    = pars['kmin']
    kmax    = pars['kmax']
    pad = 0.1
    # gwf.npf.k.set_data(logk_proposal)
    
    rch_spd     = gwf.rch.stress_period_data.get_data()
    rch_spd[0]['recharge'] = rech_proposal
    gwf.rch.stress_period_data.set_data(rch_spd)
  
    fig, axes   = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)

    ax0         = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
    c           = ax0.plot_array(np.log(logk_proposal), cmap=cm.bilbao_r, alpha=1)
    divider     = make_axes_locatable(axes[0])
    cax         = divider.append_axes("right", size="3%", pad=pad)
    cbar0       = fig.colorbar(c, cax=cax)
    cbar0.mappable.set_clim(kmin, kmax)
    cbar0.set_label('Log-Conductivity (log(m/s))')
    axes[0].set_aspect('equal')
    axes[0].set_ylabel('Y-axis')
    
    ax1         = flopy.plot.PlotMapView(model=gwf, ax=axes[1])
    c           = ax1.plot_array(gwf.rch.stress_period_data.get_data()[0]['recharge'], cmap=cm.turku_r, alpha=1)
    divider     = make_axes_locatable(axes[1])
    cax1        = divider.append_axes("right", size="3%", pad=pad)
    cbar1       = fig.colorbar(c, cax=cax1)
    # cbar1.mappable.set_clim(kmin, kmax)
    cbar1.set_label('Recharge (m/s)')
    axes[1].set_aspect('equal')
    axes[1].set_ylabel('Y-axis')
    axes[1].set_xlabel('X-axis')
 
def ellipsis(cov_data, mean_cov, pars):

    center = (0, 0)  # center coordinates
    l = np.max(pars['lx'][0]*1.5)
    
    x = np.linspace(-l, l, 300)
    y = np.linspace(-l, l, 300)
    X, Y = np.meshgrid(x, y)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    for i in range(len(cov_data)):
        M = np.array(([cov_data[i]['cov_data'][0], cov_data[i]['cov_data'][1]],
                      [cov_data[i]['cov_data'][1], cov_data[i]['cov_data'][2]]))
        res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
        ax.contour(X, Y, res, levels=[0], colors='black', alpha=0.5)

    # M = np.array(([mean_cov[0], mean_cov[1]],
    #               [mean_cov[1], mean_cov[2]]))
    # res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
    # ax.contour(X, Y, res, levels=[0], colors='blue', alpha=0.5)
    
    ellipse = patches.Ellipse(center,
                              mean_cov[0]*2,
                              mean_cov[1]*2,
                              angle=np.rad2deg(mean_cov[2]),
                              fill=False,
                              color='blue')
    ax.add_patch(ellipse)
    
    ellipse = patches.Ellipse(center,
                              pars['lx'][0][0]*2,
                              pars['lx'][0][1]*2,
                              angle=np.rad2deg(pars['ang'][0]),
                              fill=False,
                              color='red')
    ax.add_patch(ellipse)
    
    # Set equal aspect ratio for the axis
    ax.set_aspect('equal', 'box')
    
    # Set axis limits
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    
    # # Add labels and title
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Ellipse')
    
    # Display the plot
    plt.grid(True)
    plt.show()

# def ellipsis(cov_data, mean_cov, pars):

#     center = (0, 0)  # center coordinates

    
#     # Create a figure and axis
#     fig, ax = plt.subplots()
    
#     for i in range(len(cov_data)):
#         ellipse = patches.Ellipse(center,
#                                   cov_data[i]['cov_data'][0],
#                                   cov_data[i]['cov_data'][1],
#                                   angle=cov_data[i]['cov_data'][2],
#                                   fill=False,
#                                   color='black',
#                                   alpha=0.5)
#         ax.add_patch(ellipse)
#     ellipse = patches.Ellipse(center,
#                               mean_cov[0],
#                               mean_cov[1],
#                               angle=mean_cov[2],
#                               fill=False,
#                               color='blue')
#     ax.add_patch(ellipse)
    
#     ellipse = patches.Ellipse(center,
#                               pars['lx'][0][0],
#                               pars['lx'][0][1],
#                               angle=pars['ang'][0],
#                               fill=False,
#                               color='red')
#     ax.add_patch(ellipse)
    
#     # Set equal aspect ratio for the axis
#     ax.set_aspect('equal', 'box')
    
#     # Set axis limits
#     ax.set_xlim(-1500, 1500)
#     ax.set_ylim(-1500, 1500)
    
#     # # Add labels and title
#     # plt.xlabel('X-axis')
#     # plt.ylabel('Y-axis')
#     # plt.title('Ellipse')
    
#     # Display the plot
#     plt.grid(True)
#     plt.show()

def movie(gwf, diff = False, bc=False, contour = False):
    
    heads = np.load('model_data/head_ref.npy')
    if diff:
        heads = heads - heads[0,:,:]

    vmin = np.min(heads)
    vmax = np.max(heads)
    fig, ax = plt.subplots(1, 1, figsize=(12,6))
    mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
    h  = mm.plot_array(heads[0, 0, :], cmap=cm.devon_r, alpha=1, vmin = vmin, vmax = vmax)
    if contour:
        mm.contour_array(heads[0, 0, :], vmin = vmin, vmax = vmax)
    plt.colorbar(h, ax=ax, label='Head [m]')  
    ax.set_aspect('equal')
     
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()  # Clear the previous plot
        ax.set_aspect('equal')
        mm = flopy.plot.PlotMapView(model=gwf, ax=ax)
        h = mm.plot_array(heads[frame], cmap=cm.devon_r, alpha=1, vmin = vmin, vmax = vmax)  # Update the plot for the current frame
        if contour:
            mm.contour_array(heads[frame, 0, :], vmin = vmin, vmax = vmax)
        # cbar.update_normal(h)  # Update colorbar
        ax.set_title(f'Time: {(frame*0.25):.2f} days')  # Optional: Add a title
        # Add any other customization you need
        
    # Create the animation
    animation = FuncAnimation(fig, update, frames=np.shape(heads)[0], interval=500, repeat=False)

    plt.close(fig)
    # Save the animation as a GIF using ffmpeg
    animation.save("Transient.gif", writer="ffmpeg", fps=36)
    
    



