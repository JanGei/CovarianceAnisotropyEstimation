import flopy
import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt

def plot_initial_k_fields(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pars,  k_f_ini: list):
    
    # assert len(k_f_ini)%2 == 0, "You should provide an even number of fields"
    k_dir = pars['k_r_d'].replace('Virtual_Reality/', '')
    kmin    = np.min(np.log(np.loadtxt(k_dir, delimiter = ',')))
    kmax    = np.max(np.log(np.loadtxt(k_dir, delimiter = ',')))
    
    
    pad = 0.1
    
    idx = np.random.randint(0, len(k_f_ini), 6)
    k_fields = [[k_f_ini[i]] for i in idx]
    
    kmin = np.min(np.log(np.array(k_fields)))
    kmax = np.max(np.log(np.array(k_fields)))
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, sharey = True, figsize=(13.5, 10.5))

    # ax0, ax2 = axes[0]
    # ax1, ax3 = axes[1]
    fs = 20
    for j in range(3):
        for i,ax in enumerate(axes[j]):
            # Plot reference field
            axf0 = flopy.plot.PlotMapView(model=gwf, ax=ax)
            c0 = axf0.plot_array(np.log(k_fields[2*j+i]), cmap=cm.bilbao_r, alpha=1, vmin=kmin, vmax=kmax)
            ax.set_aspect('equal')
            ax.tick_params(axis='x', labelsize=fs)
            ax.tick_params(axis='y', labelsize=fs)

    
    axes[0,0].set_ylabel('Y-Koordinate [m]', fontsize=fs)
    axes[1,0].set_ylabel('Y-Koordinate [m]', fontsize=fs)
    axes[2,0].set_ylabel('Y-Koordinate [m]', fontsize=fs)
    axes[2,0].set_xlabel('X-Koordinate [m]', fontsize=fs)
    axes[2,1].set_xlabel('X-Koordinate [m]', fontsize=fs)
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.06, wspace=0.15)
    # plt.title()
    fig.suptitle('Ensemble-Initialisierung mit unterschiedlichen log(K) Feldern', fontsize=fs +2, x=0.471, y=0.90, ha='center')
    # Add colorbars
    cbar0 = fig.colorbar(c0, ax=axes, fraction=0.1, pad=0.025, shrink = 0.85)
    cbar0.set_label('Log(K)', fontsize = fs-2)
    # Set custom bounds for colorbars
    cbar0.mappable.set_clim(vmin=kmin, vmax=kmax)
    cbar0.ax.tick_params(labelsize=fs)
