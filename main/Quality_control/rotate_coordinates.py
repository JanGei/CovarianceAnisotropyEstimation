import numpy as np
import sys
sys.path.append('..')
from dependencies.model_params import get
import matplotlib.pyplot as plt

pars = get()
dx = [0.05, 0.05]
nx_ex = [100, 50]
x = np.arange(-(nx_ex[0] +1) / 2 * dx[0], (nx_ex[0] - 1) / 2 * dx[0] + dx[0], dx[0])
y = np.arange(-(nx_ex[1] +1) / 2 * dx[1], (nx_ex[1] - 1) / 2 * dx[1] + dx[1], dx[1])
ang = np.deg2rad(15)

X, Y = np.meshgrid(x, y)

X2, Y2 = pars['rot2df'](X, Y, ang)

plt.figure()
plt.scatter(X,Y)
plt.scatter(X2,Y2)


