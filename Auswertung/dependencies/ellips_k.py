import numpy as np
import matplotlib.pyplot as plt
import flopy
from cmcrameri import cm
import imageio.v2 as imageio
import matplotlib.patches as patches
from gstools import krige
import gstools as gs
import os
import numpy as np
import shutil

def ellips_k(gwf, pars, cov_data, mean_cov, k_true, ppk, pp_xy, save_dir, filename_prefix='plot', movie=False):
    
    plot_dir = os.path.join(save_dir, 'plots')
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)
    
    kmin = np.min(np.log(k_true))
    kmax = np.max(np.log(k_true))
    
    mi = [10] 
    # this plot needs to be postponed untl i finally save my pilot point locations
    xyz = gwf.modelgrid.xyzcellcenters

    obsxy = pars['obsxy']
    # humin = min(min(mean_obs[:, ui[0]]), min(true_obs[:, ui[0]]))*0.98
    # humax = max(max(mean_obs[:, ui[0]]), max(true_obs[:, ui[0]]))*1.02
    # hlmin = min(min(mean_obs[:, li[0]]), min(true_obs[:, li[0]]))*0.98
    # hlmax = max(max(mean_obs[:, li[0]]), max(true_obs[:, li[0]]))*1.02
    # humin = min(min(mean_obs[:, ui[0]]), min(mean_obs[:, ui[1]]), min(true_obs[:, ui[0]]), min(true_obs[:, ui[1]]))*0.98
    # humax = max(max(mean_obs[:, ui[0]]), max(mean_obs[:, ui[1]]), max(true_obs[:, ui[0]]), max(true_obs[:, ui[1]]))*1.02
    # hlmin = min(min(mean_obs[:, li[0]]), min(mean_obs[:, li[1]]), min(true_obs[:, li[0]]), min(true_obs[:, li[1]]))*0.98
    # hlmax = max(max(mean_obs[:, li[0]]), max(mean_obs[:, li[1]]), max(true_obs[:, li[0]]), max(true_obs[:, li[0]]))*1.02
    
    l = np.max(pars['lx'][0] * 1.5)
    x = np.linspace(-600, 600, 150)
    y = np.linspace(-300, 300, 150)
    X, Y = np.meshgrid(x, y)
    
    for i in range(144):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), constrained_layout=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        ax0, ax2 = axes[0]
        ax1, ax3 = axes[1]
        fs = 18
        # :i*10:10
        
        M = np.array(([mean_cov[i*10][0], mean_cov[i*10][1]], [mean_cov[i*10][1], mean_cov[i*10][2]]))
        eigenvalues, eigenvectors = np.linalg.eig(M)
        l1, l2, angle = extract_truth(eigenvalues, eigenvectors)
        ellipse = patches.Ellipse((0, 0), l1*2, l2*2, angle=np.rad2deg(angle), fill=False, color='blue', label = 'Mittelwert', zorder = 2)
        ax0.add_patch(ellipse)

        ellipse = patches.Ellipse((0, 0), pars['lx'][0][0]*2, pars['lx'][0][1]*2, angle=pars['ang'][0], fill=False, color='red', label = 'Referenz', zorder = 2)
        ax0.add_patch(ellipse)
        
        for j, cov in enumerate(cov_data[i*10,::5]):
            M = np.array(([cov[0], cov[1]], [cov[1], cov[2]]))
            res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
            if j == mi[0]:
                eigenvalues, eigenvectors = np.linalg.eig(M)
                l1, l2, angle = extract_truth(eigenvalues, eigenvectors)
                ellipse = patches.Ellipse((0, 0), l1*2, l2*2, angle=np.rad2deg(angle), fill=False, color='green', label = 'Realisation', zorder = 2)
                ax0.add_patch(ellipse)
                m_example = M.copy()
            else:
                ax0.contour(X, Y, res, levels=[0], colors='black', alpha=0.25, zorder = 1)
            

        ax0.set_xlim(-l, l)
        ax0.set_ylim(-l, l)
        # plt.grid(True)    
        # ax0.set_aspect('equal')
        ax0.set_xlabel('Korellationslänge 1', fontsize=fs-4)
        ax0.set_ylabel('Korellationslänge 2', fontsize=fs-4)
        ax0.set_xlim([-600, 600])
        ax0.set_ylim([-300, 300])
        ax0.tick_params(axis='x', labelsize=fs-6)
        ax0.tick_params(axis='y', labelsize=fs-6)
        ax0.set_aspect('equal')
        ax0.legend(loc='upper right', fontsize = fs-8)
    
        # Plot ensemble mean field
        axf1 = flopy.plot.PlotMapView(model=gwf, ax=ax1)
        c0 = axf1.plot_array(np.log(k_true), cmap=cm.bilbao_r, alpha=1, vmin=kmin, vmax=kmax)
        ax1.set_aspect('equal')
        ax1.set_ylabel('Hochwert [m]', fontsize=fs-4)
        ax1.set_xlabel('Rechtswert [m]', fontsize=fs-4)
        ax1.tick_params(axis='x', labelsize=fs-6)
        ax1.tick_params(axis='y', labelsize=fs-6)
    
        
        eigenvalues, eigenvectors = np.linalg.eig(m_example)
        l1, l2, angle = extract_truth(eigenvalues, eigenvectors)
        
        model = gs.Matern(dim=2, var=pars['sigma'][0], angles = angle, len_scale=[l1, l2])
        krig = krige.Ordinary(model, cond_pos=(pp_xy[:,0], pp_xy[:,1]), cond_val = ppk)
        field = krig((xyz[0], xyz[1]))
        
        axf2 = flopy.plot.PlotMapView(model=gwf, ax=ax2)
        c0 = axf2.plot_array(np.log(field[0]), cmap=cm.bilbao_r, alpha=1, vmin=kmin, vmax=kmax)
        ax2.set_aspect('equal')
        ax2.set_ylabel('Hochwert [m]', fontsize=fs-4)
        ax2.set_xlabel('Rechtswert [m]', fontsize=fs-4)
        ax2.tick_params(axis='x', labelsize=fs-6)
        ax2.tick_params(axis='y', labelsize=fs-6)
        
        
        mat_dat = mean_cov[i*10]
        M = np.array(([mat_dat[0], mat_dat[1]], [mat_dat[1], mat_dat[2]]))
        eigenvalues, eigenvectors = np.linalg.eig(M)
        l1, l2, angle = extract_truth(eigenvalues, eigenvectors)
        
        model = gs.Matern(dim=2, var=pars['sigma'][0], angles = angle, len_scale=[l1, l2])
        krig = krige.Ordinary(model, cond_pos=(pp_xy[:,0], pp_xy[:,1]), cond_val = ppk)
        field = krig((xyz[0], xyz[1]))
        
        axf3 = flopy.plot.PlotMapView(model=gwf, ax=ax3)
        c0 = axf3.plot_array(np.log(field[0]), cmap=cm.bilbao_r, alpha=1, vmin=kmin, vmax=kmax)
        ax3.set_aspect('equal')
        ax3.set_ylabel('Hochwert [m]', fontsize=fs-4)
        ax3.set_xlabel('Rechtswert [m]', fontsize=fs-4)
        ax3.tick_params(axis='x', labelsize=fs-6)
        ax3.tick_params(axis='y', labelsize=fs-6)
        
        # Add colorbars
        cbar0 = fig.colorbar(c0, ax=[ax2, ax3], fraction=0.1, pad=0.09)
        cbar0.set_label('Log(K)', fontsize = fs-2)
        # Set custom bounds for colorbars
        cbar0.mappable.set_clim(vmin=kmin, vmax=kmax)
        cbar0.ax.tick_params(labelsize=fs-4)
        
        #, x=0.5, y=0.95
        fig.suptitle(f'Zeitschritt {i*10}', fontsize=fs +4, ha='center')
        ax0.set_title('Variogram Ensemble', fontsize= fs+2)
        ax1.set_title('Referenz Feld ', fontsize= fs+2)
        ax2.set_title('Realisation', fontsize= fs+2)
        ax3.set_title('Ensemble Mittelwert', fontsize= fs+2)
        
        filename = f"{filename_prefix}_{i}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        
        # plt.show()
        plt.close(fig)  # Close the figure to release memory

    if movie:
        # Create GIF
        # resolution = (1600,600)
        # Set FFmpeg parameters to set the VBV buffer size
        # ffmpeg_params = ['-bufsize', '10M', '-maxrate', '10M']  # Set the desired buffer size (e.g., 10MB)

        with imageio.get_writer(os.path.join(save_dir.replace('output', ''), 'movie.mp4'), 
                                mode='I', fps=10) as writer:
            for j in range(143):
                filename = os.path.join(plot_dir, f"{filename_prefix}_{j}.png")
                writer.append_data(imageio.imread(filename))

        # shutil.rmtree(plot_dir)
        
def extract_truth(eigenvalues, eigenvectors):
    
    lxmat = 1/np.sqrt(eigenvalues)
    
    if lxmat[0] < lxmat[1]:
        lxmat = np.flip(lxmat)
        eigenvectors = np.flip(eigenvectors, axis = 1)
    
    if eigenvectors[0,0] > 0:
        ang = np.pi/2 -np.arccos(np.dot(eigenvectors[:,0],np.array([0,1])))    

    else:
        if eigenvectors[1,0] > 0:
            ang = np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))

        else:
            ang = np.pi -np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))

    return lxmat[0], lxmat[1], ang