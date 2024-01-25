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
            
# def update(frame, head_file, ax, c):

#     c.set_array(head_file[frame, :, :])#.flatten())

#     return c,
            
# def movie(gwf, bc = False):
    
#     # ax      = flopy.plot.PlotMapView(model=gwf)
#     heads   = np.load('model_data/head_ref.npy')
#     # c       = ax.plot_array(heads[0,0,:], cmap=cm.devon_r, alpha=1)
    
#     # animation = FuncAnimation(plt.gcf(), update, fargs=(heads, ax, c), frames=np.shape(heads)[0], interval=100, blit=True)

#     # plt.show()
    
#     fig, ax = plt.subplots(1, 1)
#     fig.set_size_inches(5,5)
#     ax      = flopy.plot.PlotMapView(model=gwf)
#     c       = ax.plot_array(heads[0,0,:], cmap=cm.devon_r, alpha=1)
#     def animate(i):
#         # ax.clear()
#         # Get the point from the points list at index i
#         head = heads[i,0,:]
#         # Plot that point using the x and y coordinates
#         c.set_array(head.flatten())
#         # ax.plot_array(head, cmap=cm.devon_r, alpha=1)
#         # Set the x and y axis to display a fixed range
#         # ax.set_xlim([0, 1])
#         # ax.set_ylim([0, 1])
        
#     ani = FuncAnimation(fig, animate, frames=np.shape(heads)[0],
#                     interval=500, repeat=False)
    
#     ani.save("Transient.gif", dpi=300,
#          writer=PillowWriter(fps=12))
    
def movie(gwf, bc=False):
    # Does not work yet
    heads = np.load('model_data/head_ref.npy')

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    ax = flopy.plot.PlotMapView(model=gwf)
    c = ax.plot_array(heads[0, 0, :], cmap=cm.devon_r, alpha=1)

    def animate(i):
        head = heads[i, 0, :]
        c.set_array(head.flatten())

    ani = FuncAnimation(fig, animate, frames=np.shape(heads)[0], interval=500, repeat=False)

    # Save the animation as a GIF using imagemagick
    ani.save("Transient.gif", writer="imagemagick", fps=12, dpi=300)    



