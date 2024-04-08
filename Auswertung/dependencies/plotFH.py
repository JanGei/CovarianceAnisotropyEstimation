import numpy as np
import matplotlib.pyplot as plt
import flopy
from cmcrameri import cm
import imageio.v2 as imageio
import os
import shutil

def plot_k_fields(gwf, pars, k_fields, h_fields, true_obs, mean_obs, save_dir, filename_prefix='plot', movie=False):
    
    plot_dir = os.path.join(save_dir, 'plots')
    if os.path.exists(plot_dir):
        shutil.rmtree(plot_dir)
    os.makedirs(plot_dir)
    
    kmin = np.min(np.log(k_fields[1]))
    kmax = np.max(np.log(k_fields[1]))
    
    ui = [26, 12] #[16, 17]
    li = [3026, 1886]
    obsxy = pars['obsxy']
    # humin = min(min(mean_obs[:, ui[0]]), min(true_obs[:, ui[0]]))*0.98
    # humax = max(max(mean_obs[:, ui[0]]), max(true_obs[:, ui[0]]))*1.02
    # hlmin = min(min(mean_obs[:, li[0]]), min(true_obs[:, li[0]]))*0.98
    # hlmax = max(max(mean_obs[:, li[0]]), max(true_obs[:, li[0]]))*1.02
    h_mean = h_fields[0]
    h_true = h_fields[1].squeeze()
    h_var = h_fields[2]
    
    mg = gwf.modelgrid
    xyz = mg.xyzcellcenters
    
    humin = min(min(mean_obs[:, ui[0]]), min(mean_obs[:, ui[1]]), min(true_obs[:, ui[0]]), min(true_obs[:, ui[1]]))*0.98
    humax = max(max(mean_obs[:, ui[0]]), max(mean_obs[:, ui[1]]), max(true_obs[:, ui[0]]), max(true_obs[:, ui[1]]))*1.02
    hlmin = min(min(h_mean[:, li[0]]), min(h_mean[:, li[1]]), min(h_true[:, li[0]]), min(h_true[:, li[1]]))*0.98
    hlmax = max(max(h_mean[:, li[0]]), max(h_mean[:, li[1]]), max(h_true[:, li[0]]), max(h_true[:, li[0]]))*1.02
    
    for i in range(144):
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col', figsize=(20, 8))

        ax0, ax2 = axes[0]
        ax1, ax3 = axes[1]
        fs = 20
        # Plot reference field
        axf0 = flopy.plot.PlotMapView(model=gwf, ax=ax0)
        c0 = axf0.plot_array(np.log(k_fields[1]), cmap=cm.bilbao_r, alpha=1, vmin=kmin, vmax=kmax)
        ax0.scatter(obsxy[ui[0],0],obsxy[ui[0],1], color = "blue", marker='x')
        ax0.text(obsxy[ui[0],0]-400,obsxy[ui[0],1]-300, f'Pegel {ui[0]}', color='blue', fontsize=fs-2)
        ax0.scatter(obsxy[ui[1],0],obsxy[ui[1],1], color = "black", marker='x')
        ax0.text(obsxy[ui[1],0]-400,obsxy[ui[1],1]-300, f'Pegel {ui[1]}', color='black', fontsize=fs-2)
        ax0.set_aspect('equal')
        ax0.set_ylabel('Hochwert [m]', fontsize=fs)
        ax0.tick_params(axis='x', labelsize=fs-4)
        ax0.tick_params(axis='y', labelsize=fs-4)
    
        # Plot ensemble mean field
        axf1 = flopy.plot.PlotMapView(model=gwf, ax=ax1)
        c1 = axf1.plot_array((k_fields[0][i*10]), cmap=cm.bilbao_r, alpha=1, vmin=kmin, vmax=kmax)
        ax1.scatter(xyz[0][li[0]],xyz[1][li[0]], color = "blue", marker='x')
        ax1.text(xyz[0][li[0]]-400,xyz[1][li[0]]-300, f'Zelle A', color='blue', fontsize=fs-2)
        ax1.scatter(xyz[0][li[1]],xyz[1][li[1]], color = "black", marker='x')
        ax1.text(xyz[0][li[1]]-400,xyz[1][li[1]]-300, f'Zelle B', color='black', fontsize=fs-2)
        ax1.set_aspect('equal')
        ax1.set_ylabel('Hochwert [m]', fontsize=fs)
        ax1.set_xlabel('Rechtswert [m]', fontsize=fs)
        ax1.tick_params(axis='x', labelsize=fs-4)
        ax1.tick_params(axis='y', labelsize=fs-4)
    
        # Add colorbars
        cbar0 = fig.colorbar(c0, ax=[ax0, ax1], fraction=0.15, pad=0.09)
        cbar0.set_label('Log(K)', fontsize = fs-2)
        # Set custom bounds for colorbars
        cbar0.mappable.set_clim(vmin=kmin, vmax=kmax)
        cbar0.ax.tick_params(labelsize=fs-2)
        
        
        ax2.plot(np.arange(0,1460, 10),true_obs[::10, ui[0]], color = "blue", label = f'{ui[0]} Referenz')
        ax2.scatter(np.arange(0,i*10, 10), mean_obs[:i*10:10, ui[0]], color = "blue", marker='x', label = f'{ui[0]} Ensemble')
        ax2.plot(np.arange(0,1460, 10),true_obs[::10, ui[1]], color = "black", label = f'{ui[1]} Referenz')
        ax2.scatter(np.arange(0,i*10, 10), mean_obs[:i*10:10, ui[1]], color = "black", marker='x', label = f'{ui[1]} Ensemble')
        ax2.set_xlim([0, 1440])
        ax2.axvline(x=1200, color='r', linestyle='--')
        ax2.set_ylim([humin, humax])
        ax2.set_title('Vergleich Ensemble - Referenz (Beobachtungspegel)', fontsize = fs)
        ax2.set_ylabel('Wasserstand [m]', fontsize = fs)
        ax2.legend(loc='upper right', fontsize = fs-4)
        ax2.tick_params(axis='x', labelsize=fs-4)
        ax2.tick_params(axis='y', labelsize=fs-4)
        
        ax3.plot(np.arange(0,1460, 10),h_true[::10, li[0]], color = "blue", label = f'A Referenz')
        ax3.scatter(np.arange(0,i*10, 10), h_mean[:i*10:10, li[0]], color = "blue", marker='x', label = f'A Ensemble')
        ax3.axvline(x=1200, color='r', linestyle='--')
        ax3.plot(np.arange(0,1460, 10),h_true[::10, li[1]], color = "black", label = f'B Referenz')
        ax3.scatter(np.arange(0,i*10, 10), h_mean[:i*10:10, li[1]], color = "black", marker='x', label = f'B Ensemble')
        ax3.set_xlim([0, 1440])
        ax3.set_ylim([hlmin, hlmax])
        # ax3.set_title(f'Beispielpegel')
        ax3.set_ylabel('Wasserstand [m]', fontsize = fs)
        ax3.set_title('Vergleich Ensemble - Referenz (Beliebige Zellen)', fontsize = fs)
        ax3.set_xlabel('Zeitschritt', fontsize = fs)
        ax3.legend(loc='upper right', fontsize = fs-4)
        ax3.tick_params(axis='x', labelsize=fs-4)
        ax3.tick_params(axis='y', labelsize=fs-4)
        
        
        fig.suptitle(f'Zeitschritt {i*10}', fontsize=fs +2, ha='center')
        ax0.set_title('Referenz Feld hydr. Leitfähigkeit', fontsize= fs)
        ax1.set_title('Ensemble Mittelwert hydr. Leitfähigkeit', fontsize= fs)
        
        filename = f"{filename_prefix}_{i}.png"
        plt.savefig(os.path.join(plot_dir, filename))
        plt.show()
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