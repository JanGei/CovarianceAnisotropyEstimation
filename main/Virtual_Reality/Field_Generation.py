# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""
import numpy as np
import os
import flopy
# from Virtual_Reality.functions.generator import gsgenerator
from dependencies.randomK_points import randomK_points
from dependencies.randomK import randomK
from scipy.interpolate import griddata
from dependencies.extract_variogram import extract_vario
# from dependencies.plotting.plot_fields import plot_fields
# import sys 

def generate_fields(pars):
    #%% Field generation (based on Olafs Skript)
    lx      = pars['lx']
    ang     = pars['ang']
    sigma   = pars['sigma']
    sim_ws = pars['sim_ws']
    mname = pars['mname']
    
    sim        = flopy.mf6.modflow.MFSimulation.load(
                            version             = 'mf6', 
                            exe_name            = 'mf6',
                            sim_ws              = sim_ws, 
                            verbosity_level     = 0
                            )
    
    gwf = sim.get_model(mname)
    mg = gwf.modelgrid
    cxy = np.vstack((mg.xyzcellcenters[0], mg.xyzcellcenters[1])).T
    # dxmax      = np.max([max(sublist) - min(sublist) for sublist in mg.xvertices])
    # dymax      = np.max([max(sublist) - min(sublist) for sublist in mg.yvertices])
    # dx         = [dxmax, dymax]
   
    #%% Field generation
    # Kflat, K  = randomK_points(mg.extent, cxy, dx,  lx[0], -np.deg2rad(ang[0]), np.exp(sigma[0]), pars, random = False, ftype = 'K')
    # Rflat, R  = randomK_points(mg.extent, cxy, dx,  lx[1], -np.deg2rad(ang[1]), sigma[1], pars, random = False, ftype = 'R')
    x = np.arange(pars['dx'][0]/2, pars['nx'][0]*pars['dx'][0], pars['dx'][0])
    y = np.arange(pars['dx'][1]/2, pars['nx'][1]*pars['dx'][1], pars['dx'][1])

    # Grid in Physical Coordinates
    X, Y = np.meshgrid(x, y)
    
    K = randomK(np.deg2rad(ang[0]), sigma[0], pars['cov'], 1, pars, ftype = 'K', random = False)
    # extract_vario(X.ravel(order = 'F'), Y.ravel(order = 'F'), K.ravel(order = 'F')) #not functional.. yet)
    R = randomK(np.deg2rad(ang[1]), sigma[1], pars['cov'], 1, pars, ftype = 'R', random = False)
    # Anmerkung des Übersetzers: Beim generieren dieser Felder ist die Varianz per se dimensionslos
    # Wenn wir also die Felder von Erdal und Cirpka nachbilden wollen, müssen wir überhaupt nicht
    # die Varianz mitscalieren, wenn die Einheiten geändert werden, sonder nur der mean
    
    Kflat =  griddata((X.ravel(order = 'F'), Y.ravel(order = 'F')), K.ravel(order = 'F'),
                     (cxy[:,0], cxy[:,1]), method='nearest')
    Rflat =  griddata((X.ravel(order = 'F'), Y.ravel(order = 'F')), R.ravel(order = 'F'),
                     (cxy[:,0], cxy[:,1]), method='nearest')
    #%% Saving the fields - Übergabe in (m/s)
    np.savetxt(os.path.join(pars['k_r_d']), Kflat, delimiter = ',')
    np.savetxt(os.path.join(pars['r_r_d']), Rflat/1000/86400, delimiter = ',')

    # return Kflat, Rflat/1000/86400, K, R