import numpy as np

    
    
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