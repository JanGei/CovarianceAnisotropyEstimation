import matplotlib.pyplot as plt
import numpy as np
import cmath 

# def solve_quadratic(a, b, c):
#     # Calculate the discriminant
#     discriminant = (b**2) - (4*a*c)

#     # Find two solutions
#     sol1 = (-b - cmath.sqrt(discriminant)) / (2 * a)
#     sol2 = (-b + cmath.sqrt(discriminant)) / (2 * a)

#     return sol1, sol2

# a = 2
# m = 1
# b = 6

# mat = np.array(([a, m],[m,b]))

# eig1, eig2 = solve_quadratic(1,-(mat[0,0]+mat[1,1]), mat[0,0]*mat[1,1]-mat[1,0]*mat[0,1])

# # Die Eigenwerte sind die inversen radien in den Hauptrichtungen zum Quadrat
# eigenvalues, eigenvectors = np.linalg.eig(mat)

# # m_a = 1
# # m_b = -(a+b)
# # m_c = a*b-m**2

# # test1, test2 = solve_quadratic(m_a, m_b, m_c)

# lambda1 = (a+b - np.sqrt( (a+b)**2 -4*(a*b-m**2)))/2
# lambda2 = (a+b + np.sqrt( (a+b)**2 -4*(a*b-m**2)))/2 

# # calculate eigenvectors
# rhs = np.array([0,0])
# x1,x2 = np.linalg.solve(mat-np.eye(2)*lambda1, rhs)


# e1 = np.array([(a-b-np.sqrt((a-b)**2-4*m**2))/(2*m),1])
# e2 = np.array([(a-b+np.sqrt((a-b)**2-4*m*+2))/(2*m),1])


# Initialisierung einer Ellipse:
l1 = 2000
l2 = 600
ang = np.deg2rad(np.array(90))

D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 

M = D @ np.array([[1/l1**2, 0],[0, 1/l2**2]]) @ D.T

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(M)

# Get the major and minor semi-axes lengths
a = 1 / np.sqrt(eigenvalues[0])
b = 1 / np.sqrt(eigenvalues[1])

# Get the rotation angle
angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

# Plot the ellipse
theta = np.linspace(0, 2*np.pi, 100)
ellipse_x = a * np.cos(theta)
ellipse_y = b * np.sin(theta)

# Rotate the ellipse
rot_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
rotated_ellipse = np.dot(rot_matrix, np.array([ellipse_x, ellipse_y])) 

# Matrix Ansatz
l = np.max([l1,l2])
x = np.linspace(-l*2, l*2, 400)
y = np.linspace(-l*2, l*2, 400)
X, Y = np.meshgrid(x, y)

# Compute the result of the equation for each point
result = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1

# Plot the ellipse
plt.figure(figsize=(9, 9))
plt.contour(X, Y, result, levels=[0], colors='r')
# plt.plot(rotated_ellipse[0], rotated_ellipse[1], 'k')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Ellipse Plot')
plt.show()
    
