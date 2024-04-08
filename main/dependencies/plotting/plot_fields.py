import flopy
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_fields(gwf: flopy.mf6.modflow.mfgwf.ModflowGwf, pars,  logk_proposal, rech_proposal: np.ndarray):
    
    kmin    = np.min(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    kmax    = np.max(np.log(np.loadtxt(pars['k_r_d'], delimiter = ',')))
    
    pad = 0.1
    # gwf.npf.k.set_data(logk_proposal)
    
    rch_spd     = gwf.rch.stress_period_data.get_data()
    rch_spd[0]['recharge'] = rech_proposal
    gwf.rch.stress_period_data.set_data(rch_spd)
  
    fig, axes   = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)

    ax0         = flopy.plot.PlotMapView(model=gwf, ax=axes[0])
    c           = ax0.plot_array(logk_proposal, cmap=cm.bilbao_r, alpha=1)
    divider     = make_axes_locatable(axes[0])
    cax         = divider.append_axes("right", size="3%", pad=pad)
    cbar0       = fig.colorbar(c, cax=cax)
    cbar0.mappable.set_clim(kmin, kmax)
    cbar0.set_label('Log-Conductivity (log(m/s))')
    axes[0].set_aspect('equal')
    axes[0].set_ylabel('Y-axis')
    
    ax1         = flopy.plot.PlotMapView(model=gwf, ax=axes[1])
    c           = ax1.plot_array(gwf.rch.stress_period_data.get_data()[0]['recharge'], cmap=cm.turku_r, alpha=1)
    divider     = make_axes_locatable(axes[1])
    cax1        = divider.append_axes("right", size="3%", pad=pad)
    cbar1       = fig.colorbar(c, cax=cax1)
    # cbar1.mappable.set_clim(kmin, kmax)
    cbar1.set_label('Recharge (m/s)')
    axes[1].set_aspect('equal')
    axes[1].set_ylabel('Y-axis')
    axes[1].set_xlabel('X-axis')
    
    plt.show()