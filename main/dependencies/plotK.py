import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from matplotlib.colors import Normalize

def plot_K(nx, dx, K, colorlimits = [], points = [], pval = []):
    # Create meshgrid
    X, Z = np.meshgrid(np.arange(0, (nx[0]+1)*dx[0], dx[0]), 
                       np.arange(0, (nx[1]+1)*dx[1], dx[1]))

    # Plot the data - CHECK THIS!!
    plotK = np.hstack((K, K[:,-1].reshape(len(K[:,-1]),1)))
    k_add = np.hstack((K[-1,:], K[-1,-1]))
    plotK = np.vstack((plotK, k_add))    
    

    plt.pcolor(X, Z, np.log(plotK), cmap = cm.bilbao_r)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    cb = plt.colorbar()
    
    if len(points) != 0:
        norm = Normalize(vmin=np.min(plotK), vmax=np.max(plotK))
        indmeas = np.floor(points @ np.linalg.inv(np.diag(dx)))
        maxdif = 0
        meandif = 0
        for ii in range(len(points)):
            plt.scatter(points[ii, 0], points[ii, 1], c=pval[ii], cmap = cm.bilbao_r, s=50, edgecolors='black', linewidths=1.5, norm=norm)
            dif = abs(np.log(K[int(indmeas[ii,0]), int(indmeas[ii,1])]) - pval[ii])
            maxdif = np.max([maxdif, dif[0]])
            meandif += dif
        print(maxdif, meandif/len(points))
    
    if len(colorlimits) != 0:
        plt.clim(colorlimits)
        # cb = plt.colorbar()
        cb.set_ticks(np.arange(np.ceil(colorlimits[0]), np.floor(colorlimits[1]) + 1))
        cb.set_ticklabels(['$10^{{{}}}$'.format(int(i)) for i in np.arange(np.ceil(colorlimits[0]), np.floor(colorlimits[1]) + 1)])
        cb.set_label('K [m/s]')
    else:
        minOM = np.ceil(np.min(plt.gci().get_clim()))
        maxOM = np.floor(np.max(plt.gci().get_clim()))
        ytick = np.arange(minOM, maxOM+1)

        yticklabels = [r'$10^{{{}}}$'.format(int(i)) for i in ytick]
        cb.set_ticks(ytick)
        cb.set_ticklabels(yticklabels)
        cb.set_label('K [m/s]')
        colorbar_limits = cb.mappable.get_clim()
        plt.title('Reference Field')
        
        return colorbar_limits
        
    
    
