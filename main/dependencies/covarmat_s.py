import numpy as np



def covarmat_s(Xint, Xmeas, pars, theta):
    # Xint: (n x dim) array of interpolation locations
    # Xmeas: (m x dim) array of measurement locations
    # Ctype: type of covariance model (1: exponential, 2: Gaussian, 3: Matern)
    # theta: (n_theta x 1) array of structural parameters
    #        first: variance
    #        following: correlation lengths
    #        if only one corr. length, assume isotropy
    
    Ctype = pars['cov']
    m, dim = Xmeas.shape
    n = Xint.shape[0]
    # ang = theta[2]
    sigma2 = np.array(theta[0])
    lx = np.array(theta[1]).flatten()
    if len(lx) == 1:
        lx = np.tile(lx, dim)
    
    rotmat = pars['rotmat'](-theta[2])
    # How to fix this?
    Xint = np.dot(rotmat, Xint.T).T
    Xmeas = np.dot(rotmat, Xmeas.T).T
    
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
    
    Q_ssm = pars['covmat'](H, sigma2, Ctype)

    return Q_ssm