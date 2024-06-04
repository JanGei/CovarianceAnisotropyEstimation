import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
# from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
# from plot_angles import plot_angles
from dependencies.randomK import randomK
# from dependencies.create_k_fields import create_k_fields
# from dependencies.create_pilot_points import create_pilot_points

# from dependencies.randomK import randomK
# import matplotlib.pyplot as plt
from dependencies.plotK import plot_K
from dependencies.plot_measurements_2D import plot_measurements_2D
from dependencies.covarmat_s import covarmat_s
'''
    Example of Generating Conditional Realizations 
    Generates a random field, takes measurements, adds noise to the
    measurments, and generates conditional realizations meeting the 
    measurements.
'''

if __name__ == '__main__':
    #%% Step 1: Generating a random field
    pars = get()
    # true parameters
    nx = np.array([100,100]) # number of cells
    dx = np.array([1,1])     # grid spacing [m]
    lx = np.array([10,3])   # isotropic correlation length/range [m]
    sigY = np.array(2)       # variance of log-conductivity
    Kg = np.array([1e-4])    # geometric mean
    angles = np.deg2rad(np.arange(20, 160, 10))     # rotation angle in ° counter clockwise from positive x-axis

    plt.figure(figsize=(10, 8))
    K = randomK(angles[0], sigY, pars['cov'], Kg, grid = [nx, dx, lx * np.array([1, 1])], pars = pars)
    colorlimits  = plot_K(nx,dx,K)
    
    for ang in angles:
        K = randomK(ang, sigY, pars['cov'], Kg, grid = [nx, dx, lx * np.array([1, 1])], pars = pars)

        
        # %% Step 2: Take measurements at randomly chosen locations and add random noise
        
        # random measurement locations
        m = 100   # number of locations
        Xmeas = np.random.uniform(0,1,(m,2)) @ np.diag(nx*dx)# measurement locations
        # Xmeas =  np.column_stack((np.linspace(0.01, 0.99, 100), np.linspace(0.01, 0.99, 100))) @ np.diag(nx*dx)
        sig_meas = 0.1 # standard deviation of measurement error
            
        # pick the values
        indmeas = np.floor(Xmeas @ np.linalg.inv(np.diag(dx))) # indices of the grid
        smeas   = np.zeros((m,1))
        for ii in range(m):
            smeas[ii] = np.log(K[int(indmeas[ii,1]),
                                 int(indmeas[ii,0])])
            
        
        # Add random measurement error (normal distribution with zero mean and standard
        # deviation sig_meas  and plot the measurements:
        smeas = smeas + sig_meas * np.random.randn(*smeas.shape)
        
        #%% Step 3: Conditional realizations
        
        # Interpolation points are the original grid points
        Xint, Yint = np.meshgrid(np.arange(0.5, nx[0] + 0.5) * dx[0], np.arange(0.5, nx[1] + 0.5) * dx[1])
        # Reshape Xint and Yint to form one (n x dim) matrix
        Xint_pw = np.column_stack((Xint.T.ravel(), Yint.T.ravel()))
        # Get the number of points
        n = Xint_pw.shape[0]
        
        # Construct covariance matrix of measurement error
        R = np.eye(m)* sig_meas**2
        
        # Discretized trend functions (for constant mean)
        X = np.ones((n,1))
        Xm = np.ones((m,1))
        # One = np.ones((1,n))
        
        # Construct the necessary covariance matrices
        Qssm = covarmat_s(Xint_pw,Xmeas,pars,[sigY,lx,ang])
        Qsmsm = covarmat_s(Xmeas,Xmeas,pars,[sigY,lx, ang])
            
        # kriging matrix and its inverse
        krigmat = np.vstack((np.hstack((Qsmsm+R, Xm)), np.append(Xm.T, 0)))
        ikrigmat = np.linalg.inv(krigmat)
        
        # Generate an unconditional field
        sunc = np.log(randomK(ang, sigY, pars['cov'], 1, pars, grid = [nx, dx, lx * np.array([1, 1])]))
        sunc = np.reshape(sunc,(nx[0], nx[1]))
        
        # Evaluate the unconditional field at the measurement points
        sunc_at_meas = np.zeros(m)
        for ii in range(m):
            sunc_at_meas[ii] = sunc[int(indmeas[ii, 1]), int(indmeas[ii, 0])]  
            
        # Perturb the measurements and subtract the unconditional realization
        spert = np.squeeze(smeas) + np.squeeze(sig_meas * np.random.randn(*smeas.shape)) - np.squeeze(sunc_at_meas)
        
        # Solve the kriging equation
        sol = np.linalg.lstsq(krigmat, np.append(spert.flatten(), 0), rcond=None)[0]
        
        # Separate the trend coefficient(s) from the weights of the covariance-functions in the function-estimate form
        xi = sol[:m]
        beta = sol[m]
        
        # Conditional realization of s
        s_cond = np.squeeze(Qssm.dot(xi)) + np.squeeze(X.dot(beta)) + np.squeeze(sunc.flatten())
        K_cond = np.reshape(np.exp(s_cond),(nx[1],nx[0]))
        
        plt.figure(figsize=(10, 8))
        # plotK plots logarithmic field
        plot_K(nx,dx,K_cond,colorlimits,points=indmeas, pval = smeas)
        plt.show()
