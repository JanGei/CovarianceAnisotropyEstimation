import numpy as np
import matplotlib.pyplot as plt


#%% ellipsis 1
l1 = 1487
l2 = 446
ang = 2.8251840787629807

theta = np.linspace(0, 2*np.pi, 100)
ellipse_x = l1 * np.cos(theta)
ellipse_y = l2 * np.sin(theta)

# Rotate the ellipse
rot_matrix = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
rotated_ellipse = np.dot(rot_matrix, np.array([ellipse_x, ellipse_y])) 

D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 
M = D @ np.array([[1/l1**2, 0],[0, 1/l2**2]]) @ D.T

#%% ellipsis 2
a = 8.95190282074998e-07
m = 1.352869866626093e-06
b = 4.584306866596388e-06

M = np.array(([a, m],[m,b]))

l = np.max([l1,l2])
x = np.linspace(-l*2, l*2, 400)
y = np.linspace(-l*2, l*2, 400)
X, Y = np.meshgrid(x, y)

res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1

# Plot the ellipse
plt.figure(figsize=(9, 9))
plt.contour(X, Y, res, levels=[0], colors='r')
plt.plot(rotated_ellipse[0], rotated_ellipse[1], 'k')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ellipse Plot')
plt.show()