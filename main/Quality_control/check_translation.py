import sys
sys.path.append('..')
# from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def angle_with_x_axis(x, y):
    # Calculate the slope of the line passing through the origin and the point
    slope = y / x if x != 0 else np.inf
    
    # Calculate the angle using arctangent
    angle_rad = np.arctan(slope)
    
    # Convert angle from radians to degrees
    angle_deg = np.degrees(angle_rad)
    
    # Adjust angle for different quadrants
    if x < 0:
        angle_deg += 180
    elif x >= 0 and y < 0:
        angle_deg += 360
        
    return angle_deg

pars        = get()
n_mem       = pars['n_mem']
nprocs      = pars['nprocs']
clx         = pars['lx']
# sim, gwf = load_template_model(pars)

n_target = 10000
res = np.zeros((n_target,6))

for i in range(n_target):
    lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]*2),
                    np.random.randint(pars['dx'][1], clx[0][1]*2)])
    ang = np.random.uniform(0, np.pi)
    sigma = np.random.uniform(0.5, 3)
    if lx[0] < lx[1]:
        lx = np.flip(lx)
        if ang > np.pi/2:
            ang -= np.pi/2
        else:
            ang += np.pi/2
    elif lx[0] == lx[1]:
        lx[0] += 1
    
    res[i,0:3] = lx[0], lx[1], ang
    
    D = pars['rotmat'](-ang)
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
    eigenvalues, eigenvectors =  np.linalg.eig(M)
    # res[i,3:] = extract_truth(eigenvalues, eigenvectors)
    
    res[i,3:] = pars['mat2cv'](eigenvalues, eigenvectors)

    
difference = res[:,0:3] - res[:,3:]
# difference[:,2] = difference[:,2]%np.pi
print(np.max(np.abs(difference)))

center = (0, 0)  # center coordinates
l = np.max(pars['lx'][0]*1.5)

x = np.linspace(-l, l, 300)
y = np.linspace(-l, l, 300)
X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots()
res1 = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
ax.contour(X, Y, res1, levels=[0], colors='black', alpha=0.5, zorder = 0)
X, Y = [0, 0], [0, 0]

idx = eigenvalues.argsort()  # Indices of sorted eigenvalues in ascending order
eigenvalues = eigenvalues[idx]  # Sorted eigenvalues
eigenvectors = eigenvectors[:, idx]  # Corresponding sorted eigenvectors
x = eigenvectors[0, :]
y = eigenvectors[1, :]
scale = 500
ax.plot([0, x[0]*scale], [0, y[0]*scale], color='green', linestyle='-', linewidth=2)
# ax.plot([0, -x[0]*scale], [0, -y[0]*scale], color='green', linestyle='-', linewidth=2)
ax.plot([0, x[1]*scale], [0, y[1]*scale], color='red', linestyle='-', linewidth=2)
# ax.plot([0, -x[1]*scale], [0, -y[1]*scale], color='red', linestyle='-', linewidth=2)
# ax.quiver(X, Y, x, y)

angle = angle_with_x_axis(x[0], y[0])
print("Angle with x-axis (counterclockwise):", angle, "degrees")
print("Angle 1 with atan:", np.rad2deg(np.arctan2(y[0], x[0]))%180, "degrees")
print("Angle 2 with atan:", np.rad2deg(np.arctan2(y[1], x[1]))%180, "degrees")


ellipse = patches.Ellipse(center,
                          lx[0]*2,
                          lx[1]*2,
                          angle=np.rad2deg(ang),
                          fill=False,
                          color='blue',
                          alpha = 0.5,
                          zorder = 1)
ax.add_patch(ellipse)

print(f'{np.rad2deg(ang)}')