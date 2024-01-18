# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""

import numpy as np
from cmcrameri import cm
import matplotlib.pyplot as plt
from functions.generator import gsgenerator
from functions.model_params import get
from mpl_toolkits.axes_grid1 import make_axes_locatable


#%% Field generation (based on Olafs Skript)
# Watch out as first entry corresponds to y and not to x

pars = get()
nx      = pars['nx']
dx      = pars['dx']
lx      = pars['lx']
ang     = pars['ang']
sigma   = pars['sigma']
mu      = pars['mu']
cov     = pars['cov']


#%% Field generation (based on gstools)

X,Y,logK    = gsgenerator(nx, dx, lx[0], ang[0], sigma[0],  cov, random = False) 
logK        = logK + mu[0]
X,Y,rech    = gsgenerator(nx, dx, lx[1], ang[1], sigma[1],  cov, random = False) 
rech        = rech + mu[1]


#%% plotting
cmaps = ['Blues', 'BuPu', 'CMRmap', 'Grays', 'OrRd', 'RdGy', 'YlOrBr', 'afmhot',
        'cividis', 'copper']

cmapc = ['batlowK', 'bilbao', 'berlin', 'devon', 'glasgow', 'grayC', 'lajolla',
         'lapaz', 'lipari', 'nuuk', 'oslo', 'turku']

cmnam = cm.cmaps
names = list(cmnam.keys())

windowx = np.array([0, 5000])
windowy = np.array([0, 2500])
mask_x = (X >= windowx[0]) & (X <= windowx[1])
mask_y = (Y >= windowy[0]) & (Y <= windowy[1])
mask_combined = np.ix_(mask_y[:, 0], mask_x[0, :])
Xpl = X[mask_combined]
Ypl = Y[mask_combined]

cmap_rech = cm.turku_r
cmap_logK = cm.bilbao_r

pad = 0.1

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)
ax1.pcolor(Xpl, Ypl, logK.T[mask_combined], cmap=cmap_logK)
divider = make_axes_locatable(ax1)
cax = divider.append_axes("right", size="3%", pad=pad)
cbar = fig.colorbar(ax1.pcolor(Xpl, Ypl, logK.T[mask_combined], cmap=cmap_logK), cax=cax)
cbar.set_label('Log-Conductivity (log(m/s))')
ax1.set_ylabel('Y-axis')
ax1.set_aspect('equal')

ax2.pcolor(Xpl, Ypl, rech.T[mask_combined], cmap=cmap_rech)
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="3%", pad=pad)  # Adjust the 'pad' value as needed
cbar = fig.colorbar(ax2.pcolor(Xpl, Ypl, rech.T[mask_combined], cmap=cmap_rech), cax=cax)
cbar.set_label('Log-Conductivity (log(m/s))')
ax2.set_xlabel('X-axis')
ax2.set_ylabel('Y-axis')
ax2.set_aspect('equal')

#%% Saving the fields

np.savetxt('model_data/logK_reference.csv', logK, delimiter = ',')
np.savetxt('model_data/rech_reference.csv', rech, delimiter = ',')

#%% plotting
# import pickle
# import flopy
# import numpy as np
# # import pandas as pd
# import matplotlib.pyplot as plt
# import geopandas as gpd
# from matplotlib.colors import Normalize
# from matplotlib import cm
# from flopy.utils.gridgen import Gridgen
# from flopy.discretization.structuredgrid import StructuredGrid
# from shapely.geometry import Point, LineString, shape, MultiPoint
# from gstools import Matern, krige

# import random

# def parallel_kriging(self, actual_data):

#     tasks = [self.Krigging(lay, actual_data[self.PPloc['layer'] == lay])
#              for counter, lay in enumerate(self.PPlay)
#              ]
    
#     field = np.zeros(self.model.npf.k.array.shape)
    
#     for lay, krig_result in tasks:
#         field[lay, :] = krig_result

#     self.set_kfield(field)
#     return tasks


# def Krigging(self, lay, data):

#     k = np.log(data)
    
#     pos = self.PPloc[['Rechtswert', 'Hochwert']][self.PPloc['layer'] == lay]
    
#     krig = krige.Ordinary(self.covmod, cond_pos=pos.T, cond_val=k)

#     coordinates = np.vstack([self.xc, self.yc])
    
#     return lay, krig(coordinates)[0]

        


# def get_active_cells(vgrid, idom, layer):
#     # MF_Utils - approved
#     boolean_array = idom[layer]
#     index_list = []
#     for i in range(len(boolean_array)):
#         if boolean_array[i] != 1:
#             boolean_array[i] = 0
#         else:
#             index_list.append(i)
#     return boolean_array, index_list



# model_ws = "./ModelFiles"
# model_name = "VirtualReality"

# # mf = flopy.modflow.Modflow("_temp")
# Lx = 1000.0
# Ly = 1000.0
# ztop = 100.0
# zbot = 50
# nlay = 2
# nrow = 10
# ncol = 10
# delr = np.ones(nrow)*Lx/nrow
# delc = np.ones(nrow)*Ly/ncol
# delv = (ztop - zbot) / nlay
# top =  np.ones((nrow,ncol))*ztop
# botm = np.array([np.ones((nrow,ncol))*ztop-zbot*0.5,
#                 np.ones((nrow,ncol))*(ztop-zbot)])


# strgrd = StructuredGrid(delc=delc, delr=delr, top=top, botm=botm, nlay=nlay)

# g = Gridgen(strgrd, model_ws=model_ws)

# wells = np.array([[778,217],[123,789]])
# wells_pump = [-123, -300]
# wells_lay = [1,0]

# consthead = np.array([[995.0,0], [995.0,1000]])

# # river = np.array([[0.0,30.0], [100.0,20.0], [300.0,50.0], [500.0,100.0], [750.0,300.0], [1000.0,320.0]])
# river = np.array([[0.0,330.0], [100.0,310.0], [300.0,350.0], [500.0,400.0], [750.0,600.0], [1000.0,680.0]])
# river_stages = np.array([-5,-7,-10,-12,-16,-20])+100
# riv_line = [tuple(xy) for xy in river]

# g.add_refinement_features(list(zip(wells)), "point", 4, range(nlay))
# g.add_refinement_features([riv_line], "line", 3, range(nlay))

# g.build()
# g.plot()
# disv_props = g.get_gridprops_vertexgrid()
# vgrid = flopy.discretization.VertexGrid(**disv_props)
# idom = np.ones([vgrid.nlay, vgrid.ncpl])
# strt = np.zeros([vgrid.nlay, vgrid.ncpl])+85
# ixs = flopy.utils.GridIntersect(vgrid, method = "vertex")

# ### K field
# pp1 = np.array([[50,300], [500, 220], [700,900], [200,700]])
# pp2 = np.array([[500,30], [220, 500], [900,700], [700,200]])

# k1 = np.array([1e-4, 1e-4, 3e-4, 8e-5])*86400
# k2 = np.array([1e-6, 1e-5, 3e-5, 8e-6])*86400
# K = []

# data = []
# data.append(pp1)
# data.append(k1)
# K.append(K_krig(vgrid,idom,data,0, 0, 100, T = False))
# data = []
# data.append(pp2)
# data.append(k2)
# K.append(K_krig(vgrid,idom,data,1, 0, 100, T = False))
  
# ### Recharge
# rch_avg = 3e-4
# np.random.seed(0)
# rch_arr = np.ones([vgrid.ncpl])*rch_avg
# rch_cells = np.arange(vgrid.ncpl)
# rch_lay = np.zeros(vgrid.ncpl, dtype = int)
# rch_cell2d = list(zip(rch_lay,rch_cells))
# # rch_cell2d = [rch_lay,rch_cells]
# rch_list = list(zip(rch_cell2d, rch_arr))
# for i in range(vgrid.ncpl):
#     rch_list[i] = list(rch_list[i])

# ### Constant Head Boundary

# chdLS = LineString(consthead)
# l = chdLS.length
# chd_list = []
# h1 = 70
# h2 = 65
# for i in range(len(consthead)-1):
#     chdl = LineString(np.array([consthead[i],consthead[i+1]]))
#     result = ixs.intersect(chdl)
#     for cell in result.cellids:
#         xc,yc = vgrid.xyzcellcenters[0][cell],vgrid.xyzcellcenters[1][cell]
#         chd_list.append([(0, cell), h1, abs(1e-4*86400)])
#         chd_list.append([(1, cell), h2, abs(1e-4*86400)])
        
# ### Wells
# result=ixs.intersect(MultiPoint(wells))
# well_list = []
# for i, index in zip(result.cellids, range(len(result.cellids))):
#     pump = wells_pump[index]
#     layer = wells_lay[index]
#     well_list.append([(layer,i),pump])

# ### River
# riverLS = LineString(river)
# l = riverLS.length
# riv_list = []
# h = 95
# for i in range(len(river)-1):
#     rivl = LineString(np.array([river[i],river[i+1]]))
#     result = ixs.intersect(rivl)
#     h1 = river_stages[i]
#     h2 = river_stages[i+1]
#     for cell in result.cellids:
#         xc,yc = vgrid.xyzcellcenters[0][cell],vgrid.xyzcellcenters[1][cell]
#         h -= 0.2
#         riv_list.append([(0, cell),h, abs(1e-4*86400), h-2])

# sim = flopy.mf6.MFSimulation(sim_name=model_ws, sim_ws=model_ws, verbosity_level=2)
# gwf = flopy.mf6.ModflowGwf(sim, modelname=model_name, save_flows=True)
# disv = flopy.mf6.ModflowGwfdisv(model = gwf, length_units="METERS", pname="disv",
#                                     xorigin = 0, yorigin = 0, angrot = 0, nogrb = False,
#                                     nlay = disv_props["nlay"], 
#                                     ncpl = disv_props["ncpl"],
#                                     nvert = len(disv_props["vertices"]), 
#                                     top = disv_props["top"],
#                                     botm = disv_props["botm"], 
#                                     idomain = idom, 
#                                     cell2d = disv_props["cell2d"], 
#                                     vertices = disv_props["vertices"])
# disv.export("./ModelFiles/dummy_disv.shp")
# npf = flopy.mf6.ModflowGwfnpf(model = gwf, k = K)
# tdis = flopy.mf6.ModflowTdis(
#         sim, time_units="DAYS", perioddata=[[1, 1, 1.0]],
#         # start_date_time=start_date,
#     )
# ims = flopy.mf6.ModflowIms(
#         sim,
#         print_option="SUMMARY",
#         complexity="COMPLEX",
#         linear_acceleration="BICGSTAB",
#     )
# ic = flopy.mf6.ModflowGwfic(gwf, strt = strt)

# chd = flopy.mf6.ModflowGwfghb(gwf, stress_period_data = {0:chd_list})
# rch = flopy.mf6.ModflowGwfrch(gwf, stress_period_data = {0:rch_list})
# wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data = {0:well_list})
# riv = flopy.mf6.ModflowGwfriv(gwf, stress_period_data = {0:riv_list})

# headfile = "{}.hds".format(model_name)
# head_filerecord = [headfile]
# budgetfile = "{}.cbb".format(model_name)
# budget_filerecord = [budgetfile]
# saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
# printrecord = [("HEAD", "LAST")]

# oc = flopy.mf6.ModflowGwfoc(
# gwf,
# saverecord=saverecord,
# head_filerecord=head_filerecord,
# budget_filerecord=budget_filerecord,
# printrecord=printrecord
# )

# sim.write_simulation()
# sim.run_simulation()

# disv_shp = gpd.read_file("./ModelFiles/dummy_disv.shp")
# heads = flopy.utils.binaryfile.HeadFile("./ModelFiles/"+headfile).get_data(kstpkper=(0, 0))
# for lay in range(vgrid.nlay):
#     fig = plt.figure(figsize=(18, 12))
#     # ax = fig.add_subplot(1, 1, 1, aspect='equal')
#     disv_shp[f"heads_l{lay}"] = heads[lay][0]
#     ax = disv_shp.plot(column = f"heads_l{lay}")
#     norm = Normalize(vmin=disv_shp[f"heads_l{lay}"][disv_shp[f"heads_l{lay}"]>0].min(), vmax=disv_shp[f"heads_l{lay}"].max())
#     n_cmap = cm.ScalarMappable(norm=norm)
#     n_cmap.set_array([])
#     ax.get_figure().colorbar(n_cmap, label = "h [m a.s.l.]", ax = plt.gca())
#     plt.axis('equal')
#     plt.title(f"Hydraulic head in layer {lay+1}")
#     plt.show()
