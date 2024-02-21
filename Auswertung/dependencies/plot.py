#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:54:48 2024

@author: janek
"""
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import numpy as np
import shutil
import matplotlib.patches as patches



def ellipsis(cov_data, mean_cov, pars, save_dir, filename_prefix='ellipsis_plot', movie = False):
    center = (0, 0)  # center coordinates
    plot_dir = os.path.join(save_dir, 'plots')
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
        os.makedirs(plot_dir)
    else:
        os.makedirs(plot_dir)
    
    center = (0, 0)  # center coordinates
    l = np.max(pars['lx'][0]*1.5)
    
    x = np.linspace(-l, l, 300)
    y = np.linspace(-l, l, 300)
    X, Y = np.meshgrid(x, y)
    

    
    for j in range(len(mean_cov)):
        fig, ax = plt.subplots(figsize=(9,9))
        
        for i in range(len(cov_data[1])):
            M = np.array(([cov_data[j,i,0], cov_data[j,i,0]],
                          [cov_data[j,i,0], cov_data[j,i,0]]))
            
            res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
            ax.contour(X, Y, res, levels=[0], colors='black', alpha=0.5)
            

        M = np.array(([mean_cov[j][0], mean_cov[j][1]],
                      [mean_cov[j][1], mean_cov[j][2]]))
        res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
        ax.contour(X, Y, res, levels=[0], colors='blue', alpha=0.5)
        
        
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
        





    



