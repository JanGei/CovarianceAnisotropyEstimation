import numpy as np
import matplotlib.pyplot as plt

def plot_measurements_2D(Xmeas, smeas, xlimits, ylimits, colorlimits):
    m = len(smeas)
    smeas = np.log10(smeas)
    cmap = plt.get_cmap('jet')  # Get the colormap object
    norm = plt.Normalize(vmin=colorlimits[0], vmax=colorlimits[1])
    map = cmap(norm(smeas))
    
    for ii in range(m):
        plt.scatter(Xmeas[ii, 0], Xmeas[ii, 1], c=map[ii])
    
    plt.clim(colorlimits)
    plt.xlim(xlimits)
    plt.ylim(ylimits)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Measurements')
    
    # Create a dummy image plot for colorbar
    img = plt.imshow([[colorlimits[0], colorlimits[1]]], cmap=cmap, norm=norm)
    cb = plt.colorbar(img)
    cb.set_ticks(np.arange(np.ceil(colorlimits[0]), np.floor(colorlimits[1]) + 1))
    cb.set_ticklabels(['$10^{{{}}}$'.format(int(i)) for i in np.arange(np.ceil(colorlimits[0]), np.floor(colorlimits[1]) + 1)])
    cb.set_label('K [m/s]')
    plt.show()