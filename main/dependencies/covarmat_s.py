import numpy as np



def covarmat_s(Xint, Xmeas, pars, theta):
    # Xint: (n x dim) array of interpolation locations
    # Xmeas: (m x dim) array of measurement locations
    # Ctype: type of covariance model (1: exponential, 2: Gaussian, 3: Matern)
    # theta: (n_theta x 1) array of structural parameters
    #        first: variance
    #        following: correlation lengths
    #        if only one corr. length, assume isotropy
    
    m, dim = Xmeas.shape
    n = Xint.shape[0]
    sigma2 = np.array(theta[0])
    lx = np.array(theta[1]).flatten()
    
    if len(lx) == 1:
        lx = np.tile(lx, dim)
    

    rotmat = pars['rotmat'](theta[2])
    Xint = (rotmat @ Xint.T).T
    Xmeas = (rotmat @ Xmeas.T).T
    # Xint = np.flip((rotmat @ Xint.T).T, axis = 1)
    # Xmeas = np.flip((rotmat @ Xmeas.T).T, axis = 1)

        
    # # Scaled distance between all points
    Xint_x_ex = np.reshape(Xint[:,0],(len(Xint),1)) @ np.ones((1,m))
    Xmeas_x_ex = np.ones((n,1)) @ np.reshape(Xmeas[:, 0], (len(Xmeas[:,0]),1)).T
    
    Xint_y_ex = np.reshape(Xint[:,1],(len(Xint),1)) @ np.ones((1,m))
    Xmeas_y_ex = np.ones((n,1)) @ np.reshape(Xmeas[:, 1], (len(Xmeas[:,1]),1)).T
    
    deltaXnorm = (Xint_x_ex - Xmeas_x_ex) / lx[0]
    deltaYnorm = (Xint_y_ex - Xmeas_y_ex) / lx[1]
        
    H = np.sqrt(deltaXnorm ** 2 + deltaYnorm ** 2)

    
    Q_ssm = pars['covmat'](H, sigma2, pars['cov'])

    return Q_ssm