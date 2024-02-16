

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def ellips_parametric(a,b,theta):
    
    # (x * np.cos(a) - y * np.sin(a))**2 / b + (x * np.sin(a) + y * np.cos(a))**2 
    t = np.linspace(0, 2*np.pi, 100)
    
    x_unrotated = a * np.cos(t)
    y_unrotated = b * np.sin(t)
    
    # here, we rotate anti-clockwise to match matplotlib rotation scheme
    # TODO: check how gstools is using rotation:
    # It seems that it is clockwise
    theta = np.deg2rad(-theta)
    
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    
    coordinates = np.column_stack([x_unrotated, y_unrotated])
    rotated_coordinates = np.dot(coordinates, rotation_matrix)

    return rotated_coordinates

# def ellips_parametric(a,b,theta):
    
#     # (x * np.cos(a) - y * np.sin(a))**2 / b + (x * np.sin(a) + y * np.cos(a))**2 
#     t = np.linspace(0, 2*np.pi, 100)
    
#     x_unrotated = a * np.cos(t)
#     y_unrotated = b * np.sin(t)
    
#     # here, we rotate anti-clockwise to match matplotlib rotation scheme
#     # TODO: check how gstools is using rotation:
#     # It seems that it is clockwise
#     theta = np.deg2rad(-theta)
    
#     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
#                                  [np.sin(theta), np.cos(theta)]])
    
#     coordinates = np.column_stack([x_unrotated, y_unrotated])
#     rotated_coordinates = np.dot(coordinates, rotation_matrix)

#     return rotated_coordinates

if __name__ == '__main__':
    
    lx = np.array([600, 2000])
    ang = np.array([291])+90
    center = (0,0)
    
    param_el = ellips_parametric(lx[0]/2, lx[1]/2, ang[0]+10)

    a = 2
    m = 4
    b = 6

    mat = np.array(([a, m],[m,b]))
    eigenvalues, eigenvectors = np.linalg.eig(mat)
    
    x = np.linspace(-b, b, 400)
    y = np.linspace(-b, b, 400)
    X, Y = np.meshgrid(x, y)
    
    Z = X**2*mat[0,0] + X*Y*(mat[0,1] + mat[1,0]) + Y**2*mat[1,1] - 1
    
    fig, ax = plt.subplots(figsize=(9,9))
    
    # width is the total diameter of horizontal axis
    # height is total diameter of vertical axis
    # angle is the rotation in degrees anti-clockwise
    # ellipse = patches.Ellipse(center,
    #                           lx[0],
    #                           lx[1],
    #                           angle=ang[0],
    #                           fill=False,
    #                           color='black',
    #                           alpha=0.5)

    # ax.add_patch(ellipse)
    # ax.plot(param_el[:, 0],param_el[:, 1])
    ax.contour(X, Y, Z, levels=[0], colors='r')
    ax.set_aspect('equal', 'box')
    # ax.set_xlim(-1500, 1500)
    # ax.set_ylim(-1500, 1500)
    
    plt.grid(True)
    
    # Remove axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    plt.show()