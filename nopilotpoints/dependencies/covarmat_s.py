import numpy as np

def rotate_coordinates(coordinates, ang):
    # Apply rotation to coordinates
    # rotation_matrix = np.squeeze(np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]))
    
    # rotated_coordinates = np.dot(coordinates, rotation_matrix)
    
    rc = np.ones(coordinates.shape)
    
    rc[:,0] = np.cos(ang)*coordinates[:,0] + np.sin(ang)*coordinates[:,1]
    rc[:,1] = -np.sin(ang)*coordinates[:,0] + np.cos(ang)*coordinates[:,1]
    
    return rc

def covarmat_s(Xint, Xmeas, Ctype, theta):
    # Xint: (n x dim) array of interpolation locations
    # Xmeas: (m x dim) array of measurement locations
    # Ctype: type of covariance model (1: exponential, 2: Gaussian, 3: cubic)
    # theta: (n_theta x 1) array of structural parameters
    #        first: variance
    #        following: correlation lengths
    #        if only one corr. length, assume isotropy
    
    m, dim = Xmeas.shape
    n = Xint.shape[0]
    ang = np.deg2rad(theta[2])
    sigma2 = np.array(theta[0])
    lx = np.array(theta[1]).flatten()
    if len(lx) == 1:
        lx = np.tile(lx, dim)
    
    Xint = rotate_coordinates(Xint, ang)
    Xmeas = rotate_coordinates(Xmeas, ang)
    # Scaled distance between all points
    deltaXnorm = (Xint[:, np.newaxis, 0] * np.ones((1,m)) - np.ones((n,1)) * Xmeas[:, 0]) / lx[0]
    if dim > 1:
        deltaYnorm = (Xint[:, np.newaxis, 1] - Xmeas[:, 1]) / lx[1]
        if dim == 3:
            deltaZnorm = (Xint[:, np.newaxis, 2] - Xmeas[:, 2]) / lx[2]
            H = np.sqrt(deltaXnorm ** 2 + deltaYnorm ** 2 + deltaZnorm ** 2)
        else:
            H = np.sqrt(deltaXnorm ** 2 + deltaYnorm ** 2)
    else:
        H = np.abs(deltaXnorm)

    if Ctype == 1:
        Q_ssm = sigma2 * np.exp(-H)
    elif Ctype == 2:
        Q_ssm = sigma2 * np.exp(-H ** 2)
    elif Ctype == 3:
        Q_ssm = sigma2 * (1 - 1.5 * H + 0.5 * H ** 3)
        Q_ssm[H > 1] = 0
    else:
        raise ValueError("Invalid Ctype value. Must be 1, 2, or 3.")

    return Q_ssm