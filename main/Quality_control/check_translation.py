import sys
sys.path.append('..')
# from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



pars        = get()
n_mem       = pars['n_mem']
nprocs      = pars['nprocs']
clx         = pars['lx']
# sim, gwf = load_template_model(pars)
# Example usage with your data generation loop
n_target = 1000
res = np.zeros((n_target, 6))
difference = np.zeros((len(res), 4))

# Arrays to store parameters for cases with 90° error
error_cases = []

for i in range(n_target):
    lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]*2),
                   np.random.randint(pars['dx'][1], clx[0][1]*2)])
    ang = np.random.uniform(-np.pi, np.pi)
    sigma = np.random.uniform(0.5, 3)
    
    if lx[0] == lx[1]:
        lx[0] += 1

    res[i, 0:3] = lx[0], lx[1], ang

    D = pars['rotmat'](ang)
    M = D @ np.array([[1/lx[0]**2, 0], [0, 1/lx[1]**2]]) @ D.T

    a_ext, b_ext, theta_ext = pars['mat2cv'](M)
    
    if lx[0] > lx[1]:
        if a_ext > b_ext:
            res[i, 3:] = a_ext, b_ext, theta_ext
        else:
            res[i, 3:] = b_ext, a_ext, theta_ext+np.pi/2
    else:
        if a_ext > b_ext:
            res[i, 3:] = a_ext, b_ext, theta_ext+np.pi/2
        else:
            res[i, 3:] = b_ext, a_ext, theta_ext
            
    
        
    difference[i, 0:3] = np.abs(res[i, 0:3] - res[i, 3:])
    difference[i,3] = (difference[i,2]+0.00001)%np.pi
    
    
    if difference[i,3] > 1:
        error_cases.append((lx[0], lx[1], ang, a_ext, b_ext, theta_ext, i ))



center = (0, 0)  # center coordinates


print(f"Total cases with 90° error: {len(error_cases)}")
print("Parameter combinations with 90° error:")
for idx, case in enumerate(error_cases):
    print(f"Case {idx + 1}:")
    print(f"  Index: {case[6]}")
    print(f"  Semi-major axis length: {case[0]}")
    print(f"  Semi-minor axis length: {case[1]}")
    print(f"  Rotation angle: {case[2]}")
    print(f"  Extracted semi-major axis length: {case[3]}")
    print(f"  Extracted semi-minor axis length: {case[4]}")
    print(f"  Extracted rotation angle: {case[5]}")
    print()
    
    fig, ax = plt.subplots()
    ellipse1 = patches.Ellipse(center,
                              case[0]*2,
                              case[1]*2,
                              angle=np.rad2deg(case[2]),
                              fill=False,
                              color='red',
                              alpha = 0.5,
                              zorder = 1)
    ellipse2 = patches.Ellipse(center,
                              case[3]*2,
                              case[4]*2,
                              angle=np.rad2deg(case[5]),
                              fill=False,
                              color='blue',
                              alpha = 0.5,
                              zorder = 1)
    ax.add_patch(ellipse1)
    ax.add_patch(ellipse2)
    ax.set_title(f"Case {idx + 1} with Index {case[6]}")
    l = np.max([case[0]*2, case[1]*2, case[3]*2, case[4]*2])
    ax.set_xlim([-l, l])
    ax.set_ylim([-l, l])
    plt.show()




# x = np.linspace(-l, l, 300)
# y = np.linspace(-l, l, 300)
# X, Y = np.meshgrid(x, y)

# fig, ax = plt.subplots()
# res1 = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
# ax.contour(X, Y, res1, levels=[0], colors='black', alpha=0.5, zorder = 0)
# X, Y = [0, 0], [0, 0]

# eigenvalues, eigenvectors =  np.linalg.eig(M)
# idx = eigenvalues.argsort()  # Indices of sorted eigenvalues in ascending order
# eigenvalues = eigenvalues[idx]  # Sorted eigenvalues
# eigenvectors = eigenvectors[:, idx]  # Corresponding sorted eigenvectors
# x = eigenvectors[0, :]
# y = eigenvectors[1, :]
# scale = 500
# ax.plot([0, x[0]*scale], [0, y[0]*scale], color='green', linestyle='-', linewidth=2)
# # ax.plot([0, -x[0]*scale], [0, -y[0]*scale], color='green', linestyle='-', linewidth=2)
# ax.plot([0, x[1]*scale], [0, y[1]*scale], color='red', linestyle='-', linewidth=2)
# # ax.plot([0, -x[1]*scale], [0, -y[1]*scale], color='red', linestyle='-', linewidth=2)
# # ax.quiver(X, Y, x, y)

# angle = angle_with_x_axis(x[0], y[0])
# print("Angle with x-axis (counterclockwise):", angle, "degrees")
# print("Angle 1 with atan:", np.rad2deg(np.arctan2(y[0], x[0]))%180, "degrees")
# print("Angle 2 with atan:", np.rad2deg(np.arctan2(y[1], x[1]))%180, "degrees")


# ellipse = patches.Ellipse(center,
#                           lx[0]*2,
#                           lx[1]*2,
#                           angle=np.rad2deg(ang),
#                           fill=False,
#                           color='blue',
#                           alpha = 0.5,
#                           zorder = 1)
# ax.add_patch(ellipse)

# print(f'{np.rad2deg(ang)}')