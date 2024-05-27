import sys
sys.path.append('..')
from randomK_points import randomK_points
from randomK  import randomK
import numpy as np
import matplotlib.patches as patches
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.conditional_k import conditional_k
from dependencies.create_pilot_points import create_pilot_points
import matplotlib.pyplot as plt
import time


pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy = create_pilot_points(gwf, pars)

cov     = pars['cov']
clx     = pars['lx']
angles  = pars['ang']
sigma   = pars['sigma'][0]
cov     = pars['cov']
k_ref   = np.loadtxt(pars['k_r_d'], delimiter = ',')
dx      = pars['dx']

mg = gwf.modelgrid
xyz = mg.xyzcellcenters
cxy = np.vstack((xyz[0], xyz[1])).T
sig_meas = 0.1 
factor = 1

lx = np.array([1000, 250])
ang = np.deg2rad(26)

start_time = time.time()
Kflat, K = randomK_points(mg.extent, cxy, dx,  lx, ang, sigma,  pars)
end_time = time.time()
print(f"Execution time for function1: {end_time - start_time} seconds")

start_time = time.time()
nx = np.array([int(mg.extent[1]/dx[0]), int(mg.extent[3]/dx[1])])
K2 = randomK(nx,dx,lx,ang,sigma,cov,0.1)
end_time = time.time()
print(f"Execution time for function1: {end_time - start_time} seconds")


plt.figure()
plt.imshow(np.log(K))
plt.figure()
plt.imshow(np.log(K2))