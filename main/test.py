from dependencies.randomK import randomK2, randomK
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata

lx = np.array([4250, 150])
sigY = np.array([2])
ang = np.array([0])
Ctype = 'Matern'
Kg = 1e-4
nx = np.array([300, 200])
dx = np.array([20,20])
# nx_ex = nx+np.round(5*lx/dx)
x = np.arange(-nx[0] * dx[0] / 2 + dx[0]/2, nx[0] * dx[0] / 2 , dx[0])
y = np.arange(-nx[1] * dx[1] / 2 + dx[1]/2, nx[1] * dx[1] / 2 , dx[1])
# x = np.arange(-nx[0] / 2 * dx[0], (nx[0] - 1) / 2 * dx[0] + dx[0], dx[0])
# y = np.arange(-nx[1] / 2 * dx[1], (nx[1] - 1) / 2 * dx[1] + dx[1], dx[1])

# Grid in Physical Coordinates
X, Y = np.meshgrid(x, y)

coordinates = np.vstack((X.ravel(), Y.ravel())).T

ncells  = 10000
coordinates = np.zeros((ncells, 2))
coordinates[:,0] = np.random.uniform(-nx[0] * dx[0] / 2 + dx[0]/2, nx[0] * dx[0] / 2, ncells)
coordinates[:,1] = np.random.uniform(-nx[1] * dx[1] / 2 + dx[1]/2, nx[1] * dx[1] / 2, ncells)


K = randomK(nx, dx,lx,ang,sigY,Ctype,Kg)
corval, K2 = randomK2(coordinates, dx, lx, ang, sigY, Ctype, Kg)
# 
plt.figure()
plt.pcolor(X,Y,np.log(K))
plt.gca().set_aspect('equal')

plt.figure()
# triang = Triangulation(coordinates[:,0], coordinates[:,1])
# plt.tricontourf(coordinates[:,0], coordinates[:,1], np.log(K2), levels = 100)
plt.pcolor(X,Y,np.log(K2))
plt.scatter(coordinates[:,0], coordinates[:,1], c='black', marker='x', s = 1)
plt.gca().set_aspect('equal')

print(np.mean(K))
print(np.mean(K2))
print(np.var(K))
print(np.var(K2))

if np.mean(K) > np.mean(K2):
    print('1')
else:
    print('2')
if np.var(K) > np.var(K2):
    print('a')
else:
    print('b')