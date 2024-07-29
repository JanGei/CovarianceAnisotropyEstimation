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


# %% Rotating angle
lx1 = 500
lx2 = 300

angles = np.arange(-np.pi/2, np.pi/2, 0.1)
# ang = np.radians(pars['ang'][0])

# Plot the ellipse contour
As = []
fig, ax = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig4, ax4 = plt.subplots(figsize=(10, 6))
fig5, ax5 = plt.subplots(figsize=(10, 6))
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
    ax2.scatter(np.rad2deg(ang),(A[0,0]))
    ax3.scatter(np.rad2deg(ang),(A[1,0]))
    ax4.scatter(np.rad2deg(ang),(A[1,1]))
    ax5.scatter(np.rad2deg(ang),(A[1,1]+A[0,0]))
    # print(np.exp(A[1,0]))
    # ax2.scatter(np.rad2deg(ang),A[1,0])

ax2.set_title('A[0,0]')    
ax3.set_title('A[1,0]')   
ax4.set_title('A[1,1]')   
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

#%% changing lengths
lx1 = np.arange(200, 1001, 10)
lx2 = 300

ang = np.deg2rad(17)
# ang = np.radians(pars['ang'][0])

# Plot the ellipse contour
As = []
fig, ax = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig3, ax3 = plt.subplots(figsize=(10, 6))
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax.scatter(0, 0, color='red', label='Center')  # Center of the ellipse
for j, lx in enumerate(lx1):
    D = np.array([[1/lx**2, 0], [0, 1/lx2**2]])
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
    ax2.scatter(lx,(A[0,0]))
    ax3.scatter(lx,(A[1,0]))
    ax4.scatter(lx,(A[1,1]))
    # print(np.exp(A[1,0]))
    # ax2.scatter(np.rad2deg(ang),A[1,0])

ax2.set_title('A[0,0]')    
ax3.set_title('A[1,0]')   
ax4.set_title('A[1,1]')   
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
