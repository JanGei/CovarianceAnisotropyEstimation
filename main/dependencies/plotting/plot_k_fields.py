import flopy
import matplotlib.pyplot as plt
from cmcrameri import cm
import numpy as np


def plot_k_fields(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pars,  k_fields: list, points = []):
    
    assert len(k_fields)%2 == 0, "You should provide an even number of fields"
    kmin    = np.min(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    kmax    = np.max(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    # pad = 0.1
    
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
            axf.plot_array(np.log(gwf.npf.k.array), cmap=cm.bilbao_r, alpha=1, vmin = kmin, vmax = kmax)
            ax.set_aspect('equal')
            # ax.set_title("{:.2f}".format(angle[i*2+j]))
    
    gwf.npf.k.set_data(np.mean(np.log(k_fields), axis=0))   
    ax = axes['b']
    axf = flopy.plot.PlotMapView(model=gwf, ax=ax)
    axf.plot_array((gwf.npf.k.array), cmap=cm.bilbao_r, alpha=1, vmin = kmin, vmax = kmax)
    ax.set_aspect('equal')
    plt.tight_layout()
    if len(points) > 0:
        plt.scatter(points[:,0], points[:,1], s=1, color = "black")
        
    plt.show()
