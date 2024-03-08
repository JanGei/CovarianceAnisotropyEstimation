import numpy as np
from scipy.interpolate import griddata

def rotate_coordinates(coordinates, ang):
    # Apply rotation to coordinates
    # rotation_matrix = np.squeeze(np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]))
    
    # rotated_coordinates = np.dot(coordinates, rotation_matrix)
    
    rc = np.ones(coordinates.shape)
    
    rc[:,0] = np.cos(ang)*coordinates[:,0] + np.sin(ang)*coordinates[:,1]
    rc[:,1] = -np.sin(ang)*coordinates[:,0] + np.cos(ang)*coordinates[:,1]
    
    return rc

def randomK2(coordinates, dx,  lx, ang, sigY, Ctype, Kg):
    '''
    Generate auto-correlated 2-D random hydraulic-conductivity fields using
    spectral methods.
    An extended periodic field is generated and the domain of interest is cut
    out.
    Parameters
    ----------
    coordinates: coordinates of the cell mid points
    lx (1x2)   : correlation length [m] in longitudinal/transverse direction
    ang (1x1)  : rotation angle [radians] of the covariance function
    sigY (1x1) : variance of ln(K) [K in m/s]
    Ctype (1x1): type of covariance model
              1: exponential
              2: Gaussian
              3: spherical (then lx becomes the range)
    Kg : geometric mean of hydraulic conductivity [m/s]


    Returns
    K          : Field of hydraulic Conductivity

    '''
    # np.random.seed(42)
    ang = np.deg2rad(ang)
    # total number of nodes
    ntot = len(coordinates)
    
    ref = 1
    dx = dx/ref
    
    xmax = np.max(coordinates[:,0])
    ymax = np.max(coordinates[:,1])
    xmin = np.min(coordinates[:,0])
    ymin = np.min(coordinates[:,1])
    
    nx = int(np.ceil((xmax - xmin)/dx[0]))
    ny = int(np.ceil((ymax - ymin)/dx[1]))
    
    nx_ex = np.round((np.array([xmax-xmin, ymax-ymin]) + 5*lx) /dx).astype(int)
    
    x = np.arange(-nx_ex[0] / 2 * dx[0], (nx_ex[0] - 1) / 2 * dx[0] + dx[0], dx[0])
    y = np.arange(-nx_ex[1] / 2 * dx[1], (nx_ex[1] - 1) / 2 * dx[1] + dx[1], dx[1])
    
    # Grid in Physical Coordinates
    X, Y = np.meshgrid(x, y)
    
    # Rotation into Longitudinal/Transverse Coordinates
    X2 = np.cos(ang)*X + np.sin(ang)*Y
    Y2 = -np.sin(ang)*X + np.cos(ang)*Y
    
    # Scaling by correlation lengths
    H = np.sqrt((X2/lx[0])**2+(Y2/lx[1])**2)
    
    # Covariance Matrix of Log-Conductivities
    if Ctype == 'Exponential': # Exponential
        RYY = sigY * np.exp(-abs(H))
    elif Ctype == 'Matern': # Matern 3/2
       RYY = sigY * np.multiply((1+np.sqrt(3)*H), np.exp(-np.sqrt(3)*H))
    elif Ctype == 'Gaussian': # Gaussian
       RYY = sigY * np.exp(-H**2)

       
    # ============== END AUTO-COVARIANCE BLOCK ================================
    
    # ============== BEGIN POWER-SPECTRUM BLOCK ===============================
    # Fourier Transform (Origin Shifted to Node (1,1))
    # Yields Power Spectrum of the field
    SYY = np.fft.fftn(np.fft.fftshift(RYY)) / ntot
    # Remove Imaginary Artifacts
    SYY = np.abs(SYY)
    SYY[0,0] = 0
    # ============== END POWER-SPECTRUM BLOCK =================================
    
    # ============== BEGIN FIELD GENERATION BLOCK =====================================
    # Generate a field of random real numbers,
    # transform the field into the spectral domain,
    # and evaluate the corresponding phase-spectrum.
    # This random phase-spectrum and the given power spectrum
    # define the Fourier transform of the random autocorrelated field.
    shape = np.shape(SYY)
    ran = np.multiply(np.sqrt(SYY),
                      (np.random.randn(shape[0], shape[1]) +
                       1j * np.random.randn(shape[0], shape[1])))

    # Backtransformation into the physical coordinates
    K = Kg * np.exp(np.real(np.fft.ifftn(ran*ntot)))
    K = K[0:ny, 0:nx]
    
    # generating an associated grid for K 
    x = np.linspace(xmin, xmax, K.shape[0])
    y = np.linspace(ymin, ymax, K.shape[1])
    x_grid, y_grid = np.meshgrid(x,y)
    values_at_coordinates = griddata((x_grid.ravel(), y_grid.ravel()), K.ravel(),
                                     (coordinates[:,0], coordinates[:,1]), method='nearest')
    
    return values_at_coordinates, K
    
    
    
    
    
def randomK(nx,dx,lx,ang,sigY,Ctype,Kg):
    '''
    Generate auto-correlated 2-D random hydraulic-conductivity fields using
    spectral methods.
    An extended periodic field is generated and the domain of interest is cut
    out.
    Parameters
    ----------
    nx (1x2)   : number of cells in x- and y-direction
    dx (1x2)   : grid spaceing [m] in x- and y-direction
    lx (1x2)   : correlation length [m] in longitudinal/transverse direction
    ang (1x1)  : rotation angle [radians] of the covariance function
    sigY (1x1) : variance of ln(K) [K in m/s]
    Ctype (1x1): type of covariance model
              1: exponential
              2: Gaussian
              3: spherical (then lx becomes the range)
    Kg : geometric mean of hydraulic conductivity [m/s]


    Returns
    K          : Field of hydraulic Conductivity

    '''
    # np.random.seed(42)
    nx_ex = nx + np.round(5*lx/dx)
    ang = np.deg2rad(ang)
    
    # total number of nodes
    ntot = np.prod(nx_ex)
    
    # Define the physical grid
    # Grid in Physical Coordinates
    x = np.arange(-nx_ex[0] / 2 * dx[0], (nx_ex[0] - 1) / 2 * dx[0] + dx[0], dx[0])
    y = np.arange(-nx_ex[1] / 2 * dx[1], (nx_ex[1] - 1) / 2 * dx[1] + dx[1], dx[1])

    # Grid in Physical Coordinates
    X, Y = np.meshgrid(x, y)
    
    # Rotation into Longitudinal/Transverse Coordinates
    X2 = np.cos(ang)*X + np.sin(ang)*Y
    Y2 = -np.sin(ang)*X + np.cos(ang)*Y
    
    # Scaling by correlation lengths
    H = np.sqrt((X2/lx[0])**2+(Y2/lx[1])**2)
    
    # Covariance Matrix of Log-Conductivities
    if Ctype == 'Exponential': # Exponential
        RYY = sigY * np.exp(-abs(H))
    elif Ctype == 'Matern': # Matern 3/2
       RYY = sigY * np.multiply((1+np.sqrt(3)*H), np.exp(-np.sqrt(3)*H))
    elif Ctype == 'Gaussian': # Gaussian
       RYY = sigY * np.exp(-H**2)

       
    # ============== END AUTO-COVARIANCE BLOCK ================================
    
    # ============== BEGIN POWER-SPECTRUM BLOCK ===============================
    # Fourier Transform (Origin Shifted to Node (1,1))
    # Yields Power Spectrum of the field
    SYY = np.fft.fftn(np.fft.fftshift(RYY)) / ntot
    # Remove Imaginary Artifacts
    SYY = np.abs(SYY)
    SYY[0,0] = 0
    # ============== END POWER-SPECTRUM BLOCK =================================
    
    # ============== BEGIN FIELD GENERATION BLOCK =====================================
    # Generate a field of random real numbers,
    # transform the field into the spectral domain,
    # and evaluate the corresponding phase-spectrum.
    # This random phase-spectrum and the given power spectrum
    # define the Fourier transform of the random autocorrelated field.
    shape = np.shape(SYY)
    ran = np.multiply(np.sqrt(SYY),
                      (np.random.randn(shape[0], shape[1]) +
                       1j * np.random.randn(shape[0], shape[1])))

    # Backtransformation into the physical coordinates
    K = Kg * np.exp(np.real(np.fft.ifftn(ran*ntot)))
    K = K[0:nx[1], 0:nx[0]]
    
    return K