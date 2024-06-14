import numpy as np
import matplotlib.pyplot as plt
# import flopy
# from cmcrameri import cm
# import imageio.v2 as imageio
# import os
# import shutil
# import cv2

def check_observations(true_obs, mean_obs, true_h, mean_h):
    # only pass all data up until that point

    np.random.seed(42)
    obs_ind = [np.random.randint(0, len(true_obs[1])) for _ in range(4)]
    cell_ind = [np.random.randint(0, len(true_h[1])) for _ in range(4)]
    
    obs1min = min(min(mean_obs[:, obs_ind[0]]), min(mean_obs[:, obs_ind[1]]), min(true_obs[:, obs_ind[0]]), min(true_obs[:, obs_ind[1]]))*0.98
    obs1max = max(max(mean_obs[:, obs_ind[0]]), max(mean_obs[:, obs_ind[1]]), max(true_obs[:, obs_ind[0]]), max(true_obs[:, obs_ind[1]]))*1.02
    obs2min = min(min(mean_obs[:, obs_ind[2]]), min(mean_obs[:, obs_ind[3]]), min(true_obs[:, obs_ind[2]]), min(true_obs[:, obs_ind[3]]))*0.98
    obs2max = max(max(mean_obs[:, obs_ind[2]]), max(mean_obs[:, obs_ind[3]]), max(true_obs[:, obs_ind[2]]), max(true_obs[:, obs_ind[3]]))*1.02
    
    cel1min = min(min(mean_h[:, obs_ind[0]]), min(mean_h[:, obs_ind[1]]), min(true_h[:, obs_ind[0]]), min(true_h[:, obs_ind[1]]))*0.98
    cel1max = max(max(mean_h[:, obs_ind[0]]), max(mean_h[:, obs_ind[1]]), max(true_h[:, obs_ind[0]]), max(true_h[:, obs_ind[1]]))*1.02
    cel2min = min(min(mean_h[:, obs_ind[2]]), min(mean_h[:, obs_ind[3]]), min(true_h[:, obs_ind[2]]), min(true_h[:, obs_ind[3]]))*0.98
    cel2max = max(max(mean_h[:, obs_ind[2]]), max(mean_h[:, obs_ind[3]]), max(true_h[:, obs_ind[2]]), max(true_h[:, obs_ind[3]]))*1.02
    
    

    nsteps = len(mean_obs)

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col', figsize=(20, 8))

    ax0, ax2 = axes[0]
    ax1, ax3 = axes[1]
    fs = 20
    
    ax0.plot(np.arange(0,nsteps, 10),true_obs[::10, obs_ind[0]], color = "blue")
    ax0.scatter(np.arange(0,nsteps, 10), mean_obs[::10, obs_ind[0]], color = "blue", marker='x')
    ax0.plot(np.arange(0,nsteps, 10),true_obs[::10, obs_ind[1]], color = "black")
    ax0.scatter(np.arange(0,nsteps, 10), mean_obs[::10, obs_ind[1]], color = "black", marker='x')
    ax0.set_xlim([0, nsteps])
    # ax0.axvline(x=1200, color='r', linestyle='--')
    ax0.set_ylim([obs1min, obs1max])
    # ax0.set_title(f'{labels[6,x]}', fontsize = fs)
    # ax0.set_ylabel(f'{labels[8,x]} [m]', fontsize = fs)
    ax0.legend(loc='upper right', fontsize = fs-4)
    ax0.tick_params(axis='x', labelsize=fs-4)
    ax0.tick_params(axis='y', labelsize=fs-4)
    
    ax1.plot(np.arange(0,nsteps, 1),true_obs[::nsteps, obs_ind[2]], color = "blue")
    ax1.scatter(np.arange(0,nsteps, 10), mean_obs[::10, obs_ind[2]], color = "blue", marker='x')
    ax1.plot(np.arange(0,nsteps, 1),true_obs[::nsteps, obs_ind[3]], color = "black")
    ax1.scatter(np.arange(0,nsteps, 10), mean_obs[::10, obs_ind[3]], color = "black", marker='x')
    ax1.set_xlim([0, nsteps])
    # ax1.axvline(x=1200, color='r', linestyle='--')
    ax1.set_ylim([obs2min, obs2max])
    # ax0.set_title(f'{labels[6,x]}', fontsize = fs)
    # ax0.set_ylabel(f'{labels[8,x]} [m]', fontsize = fs)
    # ax1.legend(loc='upper right', fontsize = fs-4)
    ax1.tick_params(axis='x', labelsize=fs-4)
    ax1.tick_params(axis='y', labelsize=fs-4)
    
    ax2.plot(np.arange(0,nsteps, 10),true_h[::10, cell_ind[0]], color = "blue")
    ax2.scatter(np.arange(0,nsteps, 10), mean_h[::10, cell_ind[0]], color = "blue", marker='x')
    ax2.plot(np.arange(0,nsteps, 10),true_h[::10, cell_ind[1]], color = "black")
    ax2.scatter(np.arange(0,nsteps, 10), mean_h[::10, cell_ind[1]], color = "black", marker='x')
    ax2.set_xlim([0, nsteps])
    # ax0.axvline(x=1200, color='r', linestyle='--')
    ax2.set_ylim([cel1min, cel1max])
    # ax0.set_title(f'{labels[6,x]}', fontsize = fs)
    # ax0.set_ylabel(f'{labels[8,x]} [m]', fontsize = fs)
    # ax2.legend(loc='upper right', fontsize = fs-4)
    ax2.tick_params(axis='x', labelsize=fs-4)
    ax2.tick_params(axis='y', labelsize=fs-4)
    
    ax3.plot(np.arange(0,nsteps, 10),true_h[::10, cell_ind[2]], color = "blue")
    ax3.scatter(np.arange(0,nsteps, 10), mean_h[::10, cell_ind[2]], color = "blue", marker='x')
    ax3.plot(np.arange(0,nsteps, 10),true_h[::10, cell_ind[3]], color = "black")
    ax3.scatter(np.arange(0,nsteps, 10), mean_h[::10, cell_ind[3]], color = "black", marker='x')
    ax3.set_xlim([0, nsteps])
    # ax0.axvline(x=1200, color='r', linestyle='--')
    ax3.set_ylim([cel2min, cel2max])
    # ax0.set_title(f'{labels[6,x]}', fontsize = fs)
    # ax0.set_ylabel(f'{labels[8,x]} [m]', fontsize = fs)
    # ax3.legend(loc='upper right', fontsize = fs-4)
    ax3.tick_params(axis='x', labelsize=fs-4)
    ax3.tick_params(axis='y', labelsize=fs-4)
    
    plt.show()
    plt.close(fig)  # Close the figure to release memory
