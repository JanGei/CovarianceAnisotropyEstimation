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


# original ellips
lx1 = 500
lx2 = 300
ang = np.deg2rad(45)
D = np.array([[1/lx1**2, 0], [0, 1/lx2**2]])
R = pars['rotmat'](ang)
A = R @ D @ R.T

A_1_0 = np.sin(np.linspace(0,2*np.pi,50))*3.5e-6
A_0_0 = np.sin(np.linspace(0,2*np.pi,50)+np.pi/2)*0.4e-5 + 0.6e-5
A_1_1 = np.sin(np.linspace(0,2*np.pi,50)-np.pi/2)*0.4e-5 + 0.7e-5

# plt.plot(A_0_0)
# plt.plot(A_1_0)
# plt.plot(A_1_1)


# Plot the ellipse contour
As = []

# fig2, ax2 = plt.subplots(figsize=(10, 6))
# fig3, ax3 = plt.subplots(figsize=(10, 6))
# fig4, ax4 = plt.subplots(figsize=(10, 6))

x = np.linspace(-2000, 2000, 400)
y = np.linspace(-1000, 1000, 400)
X, Y = np.meshgrid(x, y)

for i in range(3):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(0, 0, color='red', label='Center')  # Center of the ellipse
    for j in range(len(A_0_0)):
        if i == 0:
            mat = np.array([[A_0_0[j], A[1,0]],[A[1,0], A[1,1]]])   
        elif i == 1:
            mat = np.array([[A[0,0], A_1_0[j]],[A_1_0[j], A[1,1]]])
        elif i == 2:
            mat = np.array([[A[0,0], A[1,0]],[A[1,0], A_1_1[j]]])
        
        Z = mat[0, 0] * X**2 + 2 * mat[0, 1] * X * Y + mat[1, 1] * Y**2
        ax.contour(X, Y, Z, levels=[1], colors='blue')
    plt.show()
            
            
            
            
            
    # ax2.scatter(np.rad2deg(ang),(A[0,0])-1)
    # ax3.scatter(np.rad2deg(ang),(A[1,0])-1)
    # ax4.scatter(np.rad2deg(ang),(A[1,1])-1)
    # print(np.exp(A[1,0]))
    # ax2.scatter(np.rad2deg(ang),A[1,0])

# # ax2.set_title('A[0,0]')    
# # ax3.set_title('A[1,0]')   
# # ax4.set_title('A[1,1]')   
# ax.axhline(0, color='black', linewidth=0.5)
# # ax2.axhline(0, color='black', linewidth=0.5)
# # ax3.axhline(0, color='black', linewidth=0.5)
# # ax4.axhline(0, color='black', linewidth=0.5)
# ax.axvline(0, color='black', linewidth=0.5)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_title('Ellipse Defined by a Symmetric 2x2 Matrix')
# ax.grid(True)
# ax.axis('equal')
# ax.legend()    

