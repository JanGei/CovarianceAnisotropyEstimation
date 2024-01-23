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


def plot(gwf, pkgs):
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
            'cmap'      : cm.devon,
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
        if i == len(pkgs)-1:
            axes[i].set_xlabel('X-axis')


# def plot_k_rch(gwf, k, rch):
#     cmap_rech   = cm.turku_r
#     cmap_logK   = cm.bilbao_r
#     pad         = 0.1

#     # Create subplots
#     fig, axes   = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)

#     # Plot the heads using plotmapview for the first subplot
#     ax1         = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
#     c1          = ax1.plot_array(k, cmap=cmap_logK, alpha=1)
#     divider     = make_axes_locatable(axes[0])
#     cax         = divider.append_axes("right", size="3%", pad=pad)
#     cbar        = fig.colorbar(c1, cax=cax)
#     cbar.set_label('Log-Conductivity (log(m/s))')
#     axes[0].set_aspect('equal')
#     axes[0].set_ylabel('Y-axis')

#     ax2         = flopy.plot.PlotMapView(model=gwf, ax=axes[1])
#     c2          = ax2.plot_array(rch, cmap=cmap_rech, alpha=1)
#     divider     = make_axes_locatable(axes[1])
#     cax         = divider.append_axes("right", size="3%", pad=pad)
#     cbar        = fig.colorbar(c2, cax=cax)
#     cbar.set_label('Recharge (m/s))')
#     axes[1].set_aspect('equal')
#     axes[1].set_ylabel('Y-axis')
#     axes[1].set_xlabel('X-axis')
    
#     plt.tight_layout()
#     plt.show()

