import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def ellipses(mats, cov_data, mean_cov):
    
    l = np.max(cov_data[0])*2

    x = np.linspace(-l, l, 300)
    y = np.linspace(-l, l, 300)
    X, Y = np.meshgrid(x, y)
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    for i in range(len(mats)):
        M = mats[i]
        res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
        ax.contour(X, Y, res, levels=[0], colors='black', alpha=0.5)
        
    # Set equal aspect ratio for the axis
    ax.set_aspect('equal', 'box')

    # Set axis limits
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)

    # Display the plot
    plt.grid(True)
    plt.show()
    
    #%% 
    # Create a figure and axis
    center = (0, 0)  # center coordinates
    fig, ax = plt.subplots()

    for i in range(len(mats)):
        data = cov_data[i]
        ellipse = patches.Ellipse(center,
                                  data[0]*2,
                                  data[1]*2,
                                  angle=np.rad2deg(data[2]),
                                  fill=False,
                                  color='blue')
        ax.add_patch(ellipse)
    
    ellipse = patches.Ellipse(center,
                               mean_cov[0]*2,
                               mean_cov[1]*2,
                               angle=np.rad2deg(mean_cov[2]),
                               fill=False,
                               color='red')
    ax.add_patch(ellipse)    
    # Set equal aspect ratio for the axis
    ax.set_aspect('equal', 'box')

    # Set axis limits
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)

    # Display the plot
    plt.grid(True)
    plt.show()
    
    
    
mats = []
cov_data = []
# Generate random values for the upper triangular part
for i in range(24):
    lx = np.array([np.random.randint(20, 400),
                   np.random.randint(10, 250)])
    # ang = np.random.uniform(-np.pi, np.pi)
    ang = np.random.uniform(0, 0.5 * np.pi)
    
    
    D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
      
                                     
    
    
    mats.append(M)
    cov_data.append([lx[0], lx[1], ang])




placeholder = np.ones(3)
sorted_data = []
# Step 1 - allign on major axis
for i,data in enumerate(cov_data):
    if data[0] < data[1]:
        placeholder[0] = data[1]
        placeholder[1] = data[0]
        placeholder[2] = data[2] + np.pi/2
    else:
        placeholder[0] = data[0]
        placeholder[1] = data[1]
        placeholder[2] = data[2] 
        
# Step 2 - allign on upper two quadran
    placeholder[2] = placeholder[2]%np.pi

    sorted_data.append(placeholder.copy())        
    

mean_cov = np.mean(np.array(sorted_data), axis = 0)

ellipses(mats, cov_data, mean_cov)

