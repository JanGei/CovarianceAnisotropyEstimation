import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio.v2 as imageio

def ellipsis(cov_data, mean_cov, pars, save_dir, filename_prefix='ellipsis_plot', movie=False):
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








    



