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


# reconstruct original ellipse
# lx1 = clx[0][0]*0.1
# lx2 = clx[0][1]/2
lx1 = 500
lx2 = 300

angles = np.array() #np.deg2rad(np.arange(0,180,10))
# ang = np.radians(pars['ang'][0])

# Plot the ellipse contour
As = []
fig, ax = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax.scatter(0, 0, color='red', label='Center')  # Center of the ellipse
for j, ang in enumerate(angles):
    D = np.array([[1/lx1**2, 0], [0, 1/lx2**2]])
    R = pars['rotmat'](ang)
    A = R @ D @ R.T
    As.append(A)
    # Create a grid of points
    x = np.linspace(-2000, 2000, 400)
    y = np.linspace(-1000, 1000, 400)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate the quadratic form
    Z = A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2
    
    ax.contour(X, Y, Z, levels=[1], colors='blue')
    ax2.scatter(np.rad2deg(ang),(A[0,0])-1)
    ax3.scatter(np.rad2deg(ang),(A[1,0])-1)
    ax4.scatter(np.rad2deg(ang),(A[1,1])-1)
    # print(np.exp(A[1,0]))
    # ax2.scatter(np.rad2deg(ang),A[1,0])
    
ax.axhline(0, color='black', linewidth=0.5)
# ax2.axhline(0, color='black', linewidth=0.5)
# ax3.axhline(0, color='black', linewidth=0.5)
# ax4.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Ellipse Defined by a Symmetric 2x2 Matrix')
ax.grid(True)
ax.axis('equal')
ax.legend()    
plt.show()
