import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio
import flopy
from cmcrameri import cm
from matplotlib.gridspec import GridSpec

def ellipsis(cov_data, mean_cov, errors, pars, save_dir, filename_prefix='ellipsis_plot', movie=False):
    plot_dir = os.path.join(save_dir, 'plots')
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)

    l = np.max(pars['lx'][0] * 1.5)
    x = np.linspace(-l, l, 300)
    y = np.linspace(-l, l, 300)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(9, 9))
    for j, mean in enumerate(mean_cov):
        for i, cov in enumerate(cov_data[j]):
            M = np.array(([cov[0], cov[1]], [cov[1], cov[2]]))
            res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
            ax.contour(X, Y, res, levels=[0], colors='black', alpha=0.2)

        M = np.array(([mean[0], mean[1]], [mean[1], mean[2]]))
        res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
        ax.contour(X, Y, res, levels=[0], colors='blue', alpha=0.5)

        ellipse = patches.Ellipse((0, 0), pars['lx'][0][0], pars['lx'][0][1], angle=pars['ang'][0], fill=False, color='red')
        ax.add_patch(ellipse)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)
        plt.grid(True)

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'time step {j}')
        ax.text(l/4*3, l/4*3, f'OLE  {errors[j,0]} \nTE-1 {errors[j,1]}\nTE-2 {errors[j,2]}', fontsize=9, color='black')

        # Save the plot as an image
        filename = f"{filename_prefix}_{j}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        ax.clear()

    if movie:
        # Create GIF
        with imageio.get_writer(os.path.join(save_dir.replace('output', ''), 'ellipsis.gif'), mode='I') as writer:
            for j in range(len(mean_cov)):
                filename = os.path.join(plot_dir, f"{filename_prefix}_{j}.png")
                writer.append_data(imageio.imread(filename))

        shutil.rmtree(plot_dir)


def plot_k_fields(gwf, k_fields):
    
    kmin = np.min(np.log(k_fields[1]))
    kmax = np.max(np.log(k_fields[1]))
    
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex='col', sharey=True, figsize= (16,6))
    # gs = GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)

    # Plot upper left
    ax0 = axes[0]
    gwf.npf.k.set_data(k_fields[1])
    axf0 = flopy.plot.PlotMapView(model=gwf, ax=ax0)
    c0 = axf0.plot_array(np.log(gwf.npf.k.array), cmap=cm.bilbao_r, alpha=1,vmin=kmin, vmax=kmax)
    ax0.set_aspect('equal')


    # Plot upper right
    ax1 = axes[1]
    gwf.npf.k.set_data(k_fields[0])
    axf1 = flopy.plot.PlotMapView(model=gwf, ax=ax1)
    c1 = axf1.plot_array(gwf.npf.k.array, cmap=cm.bilbao_r, alpha=1,vmin=kmin, vmax=kmax)
    ax1.set_aspect('equal')


    # Plot lower
    ax2 = axes[2]
    gwf.npf.k.set_data(k_fields[0]/ np.log(k_fields[1]))
    axf2 = flopy.plot.PlotMapView(model=gwf, ax=ax2)
    c2 = axf2.plot_array((gwf.npf.k.array), cmap=cm.roma, alpha=1)
    ax2.set_aspect('equal')


    # Add colorbars
    cbar0 = fig.colorbar(c0, ax=[ax0, ax1], fraction=0.1, pad=0.01)
    cbar0.set_label('Log(K)')
    cbar1 = fig.colorbar(c2, ax=ax2, fraction=0.1, pad=0.01, aspect=10)
    cbar1.set_label('Ratio')

    
    # Set custom bounds for colorbars
    cbar0.mappable.set_clim(vmin=kmin, vmax=kmax)
    cbar1.mappable.set_clim(vmin=0.5, vmax=1.5)

    plt.show()



def ellipsis_test(cov_data, mean_cov, errors, pars):


    l = np.max(pars['lx'][0] * 1.5)
    x = np.linspace(-l, l, 300)
    y = np.linspace(-l, l, 300)
    X, Y = np.meshgrid(x, y)

    
    for j in range(120):
        fig, ax = plt.subplots(figsize=(9, 9))
        for i in range(int(len(cov_data[j])/10)):
            cov = cov_data[j*10, i*10]
            
            M = np.array(([cov[0], cov[1]], [cov[1], cov[2]]))
            res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
            ax.contour(X, Y, res, levels=[0], colors='black', alpha=0.2)
        # print(cov[0])
        
        mean = mean_cov[j*10]
        M = np.array(([mean[0], mean[1]], [mean[1], mean[2]]))
        res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
        ax.contour(X, Y, res, levels=[0], colors='blue', alpha=0.5)

        ellipse = patches.Ellipse((0, 0), pars['lx'][0][0]*2, pars['lx'][0][1]*2, angle=pars['ang'][0], fill=False, color='red')
        ax.add_patch(ellipse)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)
        plt.grid(True)

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'time step {j*10}')
        ax.text(l/4*3, l/4*3, f'OLE  {errors[j,0]} \nTE-1 {errors[j,1]}\nTE-2 {errors[j,2]}', fontsize=9, color='black')

        # Save the plot as an image
        plt.show()
        # ax.clear()




