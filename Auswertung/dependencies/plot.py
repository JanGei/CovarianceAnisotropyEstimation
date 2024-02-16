#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:54:48 2024

@author: janek
"""
import matplotlib.pyplot as plt
from cmcrameri import cm
import imageio
import os
import flopy
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil
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

    ax0          = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
    c           = ax0.plot_array(np.log(logk_proposal), cmap=cm.bilbao_r, alpha=1)
    divider     = make_axes_locatable(axes[0])
    cax         = divider.append_axes("right", size="3%", pad=pad)
    cbar0        = fig.colorbar(c, cax=cax)
    cbar0.mappable.set_clim(kmin, kmax)
    cbar0.set_label('Log-Conductivity (log(m/s))')
    axes[0].set_aspect('equal')
    axes[0].set_ylabel('Y-axis')
    
    gwf.npf.k.set_data(np.log(logk_proposal))
    ax1          = flopy.plot.PlotMapView(model=gwf, ax=axes[1])
    c           = ax1.plot_array(gwf.npf.k.array, cmap=cm.turku_r, alpha=1)
    divider     = make_axes_locatable(axes[1])
    cax         = divider.append_axes("right", size="3%", pad=pad)
    cbar1        = fig.colorbar(c, cax=cax)
    cbar1.mappable.set_clim(kmin, kmax)
    cbar1.set_label('Recharge (m/s)')
    axes[1].set_aspect('equal')
    axes[1].set_ylabel('Y-axis')
    axes[1].set_xlabel('X-axis')


def ellipsis(cov_data, mean_cov, pars, save_dir, filename_prefix='ellipsis_plot', movie = False):
    center = (0, 0)  # center coordinates
    plot_dir = save_dir + '/plots/' 
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
        os.makedirs(plot_dir)
    else:
        os.makedirs(plot_dir)
    
    
    for j in range(len(mean_cov)):
        fig, ax = plt.subplots(figsize=(9,9))
        
        for i in range(len(cov_data[1])):
            ellipse = patches.Ellipse(center,
                                      cov_data[j,i,0],
                                      cov_data[j,i,1],
                                      angle=cov_data[j,i,2],
                                      fill=False,
                                      color='black',
                                      alpha=0.5)
            ax.add_patch(ellipse)
        
        ellipse = patches.Ellipse(center,
                                  mean_cov[j][0],
                                  mean_cov[j][1],
                                  angle=mean_cov[j][2],
                                  fill=False,
                                  color='blue')
        ax.add_patch(ellipse)
        
        ellipse = patches.Ellipse(center,
                                  pars['lx'][0][0],
                                  pars['lx'][0][1],
                                  angle=pars['ang'][0],
                                  fill=False,
                                  color='red')
        ax.add_patch(ellipse)
        
        ax.set_aspect('equal', 'box')
        ax.set_xlim(-1500, 1500)
        ax.set_ylim(-1500, 1500)
        plt.grid(True)
        
        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Save the plot as an image
        filename = f"{filename_prefix}_{j}.png"
        plt.savefig(plot_dir + filename)
        plt.close()

    if movie:
        
        
        # Get the list of filenames of saved plots
        plot_files = [filename for filename in os.listdir(plot_dir) if filename.startswith(filename_prefix) and filename.endswith('.png')]
        
        # Sort the filenames in numerical order
        plot_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        
        # Create a list to store the images
        images = []
        for filename in plot_files:
            # Construct the full path to the file
            file_path = os.path.join(plot_dir, filename)
    
            # Read the image and append it to the list
            images.append(imageio.imread(file_path))
        
        # Save the images as a GIF
        imageio.mimsave(save_dir.replace('output', '') + 'ellipsis.gif', images)
        
        shutil.rmtree(plot_dir)
        




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
    
    



