import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from objects.MFModel import MFModel
from dependencies.model_params import get
from dependencies.copy import create_Test_Mod
from dependencies.load_template_model import load_template_model
from dependencies.create_pilot_points import create_pilot_points
from dependencies.intersect_with_grid import intersect_with_grid
from dependencies.create_k_fields import create_k_fields
# from plot_angles import plot_angles
from dependencies.randomK import randomK


pars = get()
sim, gwf = load_template_model(pars)
pp_cid, pp_xy, neardist = create_pilot_points(gwf, pars)
model_dir = create_Test_Mod(pars)
obs_cid = intersect_with_grid(gwf, pars['obsxy'])
pp_cid, pp_xy, near_dist = create_pilot_points(gwf, pars)
k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')

field, cor_mat, l_angs, pp, field2f = create_k_fields(gwf,
                                                      pars,
                                                      k_ref,
                                                      pp_xy,
                                                      pp_cid,
                                                      test_cov=[[1500,500], 0.17]) 

Model = MFModel(model_dir[0],
                pars,
                obs_cid,
                near_dist,
                l_angs,
                cor_mat)
Model.set_field([field], ['npf'])
Model.npf.plot()

ang = np.deg2rad(62)
lx = [750,480]
D = pars['rotmat'](ang)
M = np.matmul(np.matmul(D, np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]])), D.T)

data = [[M[0,0], M[1,0], M[1,1]],np.random.normal(-8,1.7,(len(pp_cid)))]
Model.kriging(data, pp_xy, pp_cid, [], [])
Model.npf.plot()
