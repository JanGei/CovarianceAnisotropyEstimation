import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def ellipses(cov_data, mean_cov, pars):

    center = (0, 0)  # center coordinates
    l = np.max(pars['lx'][0]*1.5)
    
    x = np.linspace(-l, l, 300)
    y = np.linspace(-l, l, 300)
    X, Y = np.meshgrid(x, y)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    for i, data in enumerate(cov_data):
        # M = np.array(([cov_data[i]['cov_data'][0], cov_data[i]['cov_data'][1]],
        #               [cov_data[i]['cov_data'][1], cov_data[i]['cov_data'][2]]))
        # res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
        # ax.contour(X, Y, res, levels=[0], colors='black', alpha=0.5, zorder = 0)
        ellipse = patches.Ellipse(center,
                                  data[0]*2,
                                  data[1]*2,
                                  angle=np.rad2deg(data[2]),
                                  fill=False,
                                  color='black',
                                  alpha = 0.5,
                                  zorder = 1)
        ax.add_patch(ellipse)

    # M = np.array(([mean_cov[0], mean_cov[1]],
    #               [mean_cov[1], mean_cov[2]]))
    # res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
    # ax.contour(X, Y, res, levels=[0], colors='blue', alpha=0.5)
    
    ellipse = patches.Ellipse(center,
                              mean_cov[0]*2,
                              mean_cov[1]*2,
                              angle=np.rad2deg(mean_cov[2]),
                              fill=False,
                              color='blue',
                              zorder = 1)
    ax.add_patch(ellipse)
    
    ellipse = patches.Ellipse(center,
                              pars['lx'][0][0]*2,
                              pars['lx'][0][1]*2,
                              angle=np.rad2deg(pars['ang'][0]),
                              fill=False,
                              color='red',
                              zorder = 2)
    ax.add_patch(ellipse)
    
    # Set equal aspect ratio for the axis
    ax.set_aspect('equal', 'box')
    
    # Set axis limits
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    
    # # Add labels and title
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Ellipse')
    
    # Display the plot
    plt.grid(True)
    plt.show()