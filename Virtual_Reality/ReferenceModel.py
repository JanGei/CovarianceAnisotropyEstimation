# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""
import flopy
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.utils.gridgen import Gridgen
from functions.model_params import get
import numpy as np


#%% Model Parameters

pars = get()
nx      = pars['nx']
dx      = pars['dx']
lx      = pars['lx']
ang     = pars['ang']
sigma   = pars['sigma']
mu      = pars['mu']
cov     = pars['cov']
toph    = pars['top']
both    = pars['bot']
nlay    = pars['nlay'][0]


model_ws    = "./model_files"
model_name  = "Reference"


#%% Grid Generation
Lx = nx[0] * dx[0]
Ly = nx[1] * dx[1]


delr = np.ones(nx[0])*Lx/nx[0]
delc = np.ones(nx[1])*Ly/nx[1]

delv = (toph - both) / nlay

top     =  np.array([np.ones((nx[1],nx[0]))]*toph)
botm    =  np.array([np.ones((nx[1],nx[0]))]*toph-both)

strgrd = StructuredGrid(delc=delc.astype(int), delr=delr.astype(int), top=top, botm=botm, nlay=nlay)

g = Gridgen(strgrd, model_ws=model_ws)


#%% Well Location
row_well    = 5
col_well    = 9
well_loc    = np.zeros((col_well*row_well,2))
for i in range(row_well):
    for j in range(col_well):
        well_loc[i*col_well + j, 0] = (20 + 10*j) #*dx[0]
        well_loc[i*col_well + j, 1] = (10 + 10*i)   #*dx[1]
        
# pumping wells should be at (5, 9, 15, 27, 31)
# CHECK WHETHER THESE WELLS ARE AT THE CORRECT LOCATION

# possible refinements
# g.add_refinement_features(list(zip(wells)), "point", 4, range(nlay))



#%% Southern Boudnary - river
river           = np.array([[0.0,0], [1000.0,0]])
river_stages    = np.array([13.4])
riv_line        = [tuple(xy) for xy in river]

# NEED VARAIBLE RIVER STAGE DATA
# possible refinements
# g.add_refinement_features([riv_line], "line", 3, range(nlay))

#%% Buildng Grid

g.build()
g.plot()
disv_props  = g.get_gridprops_vertexgrid()
vgrid       = flopy.discretization.VertexGrid(**disv_props)
idom        = np.ones([vgrid.nlay, vgrid.ncpl])
strt        = np.zeros([vgrid.nlay, vgrid.ncpl])+20
ixs         = flopy.utils.GridIntersect(vgrid, method = "vertex")

# TODO: RUN STEADYSTATE MODEL TO OBTAIN STARTING HEADS

#%% Loading reference fields
logK = np.loadtxt('model_data/logK_reference.csv', delimiter = ',')
rech = np.loadtxt('model_data/rech_reference.csv', delimiter = ',')

























# import pickle
# import flopy
# import numpy as np
# # import pandas as pd
# import matplotlib.pyplot as plt
# import geopandas as gpd
# from matplotlib.colors import Normalize
# from matplotlib import cm
# from scipy.stats import qmc


# from shapely.geometry import Point, LineString, shape, MultiPoint
# from gstools import Matern
# from pykrige.ok import OrdinaryKriging
# import random

# def kriggle(pack):
#     # MF_Utils - approved
#     # Generate regular grid
#     x,y,data,angle,Corlen = pack #[cellx,celly,data(=[xyk,k]), ...]
#     xyk, k = data
#     # Cov Model
#     model = Matern(dim = 2, var = 1.5, len_scale = Corlen, angles = angle)
#     # Kriging
#     ok1 = OrdinaryKriging(xyk[:,0], xyk[:,1], k, model)
#     z1,_ = ok1.execute("points", x, y)
#     return z1


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


# def K_krig(vgrid,idom,data,layer, angle, Corlen, T = False):
#     # Modelbuilder - approved
#     xc = vgrid.xyzcellcenters[0]
#     yc = vgrid.xyzcellcenters[1]
#     krig_pack = [xc,yc,data,angle,Corlen]
#     K_interp = kriggle(krig_pack)    
#     return K_interp




# # mf = flopy.modflow.Modflow("_temp")








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
#         riv_list.append([(0, cell),h, abs(1e-5*86400), h-2])

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