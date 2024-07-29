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

x = np.linspace(-2000, 2000, 400)
y = np.linspace(-2000, 2000, 400)
X, Y = np.meshgrid(x, y)
# sim, gwf = load_template_model(pars)
# Example usage with your data generation loop
n_target = 25
center = (0, 0)  # center coordinates
matrices = np.zeros((n_target, 4))
fig, ax = plt.subplots()

for i in range(n_target):
    lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]*2),
                   np.random.randint(pars['dx'][1], clx[0][1]*2)])
    ang = np.random.uniform(-np.pi/2, np.pi/2)
    # ang = np.random.uniform(np.deg2rad(0), np.deg2rad(45))
    # ang = np.random.uniform(np.deg2rad(45), np.deg2rad(90))
    # ang = np.random.uniform(np.deg2rad(90), np.deg2rad(135))
    # ang = np.random.uniform(np.deg2rad(135), np.deg2rad(180))
    sigma = np.random.uniform(0.5, 3)
    
    if lx[0] < lx[1]:
        lx = np.flip(lx)

    if lx[0] == lx[1]:
        lx[0] += 1


    D = pars['rotmat'](ang)
    M = D @ np.array([[1/lx[0]**2, 0], [0, 1/lx[1]**2]]) @ D.T
    
    matrices[i,:] = [M[0,0], M[1,1], M[1,1]<M[0,0], M[1,0]]
    
    Z = M[0, 0] * X**2 + 2 * M[0, 1] * X * Y + M[1, 1] * Y**2
    if M[0,0] > M[1,1]:
        color = 'blue'
    else:
        color = 'red'
    if M[1,0] < 0:
        ls = 'solid'
    else:
        ls = 'dashed'
    ax.contour(X, Y, Z, levels=[1], colors=color, linestyles = ls)
    

    



