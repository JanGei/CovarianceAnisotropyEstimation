# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""
import flopy
import matplotlib.pyplot as plt
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.utils.gridgen import Gridgen
from shapely.geometry import LineString, MultiPoint
from functions.model_params import get
from functions.plot import plot
import numpy as np


#%% Model Parameters

pars    = get()
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
mname   = pars['mname']
sname   = pars['sname']
sim_ws  = pars['sim_ws']


#%% Grid Generation
Lx = nx[0] * dx[0]
Ly = nx[1] * dx[1]


delr = np.ones(nx[0])*Lx/nx[0]
delc = np.ones(nx[1])*Ly/nx[1]

delv = (toph - both) / nlay

top     =  np.array([np.ones((nx[1],nx[0]))]*toph)
botm    =  np.array([np.zeros((nx[1],nx[0]))])

strgrd = StructuredGrid(delc=delc.astype(int), delr=delr.astype(int), top=top, botm=botm, nlay=nlay)

g = Gridgen(strgrd, model_ws=sim_ws)


#%% Well Location
welxy   = pars['welxy']
welq    = pars['welq']
welay   = pars['welay']

# possible refinements
# g.add_refinement_features(list(zip(wells)), "point", 4, range(nlay))

#%% Southern Boudnary - river
river           = pars['river']
river_stages    = pars['rivh']
rivC            = pars['rivC']
riv_line        = [tuple(xy) for xy in river]

# NEED VARAIBLE RIVER STAGE DATA
# possible refinements
# g.add_refinement_features([riv_line], "line", 3, range(nlay))

#%% Northern Boudnary - Fixed head
chdl            = pars['chd']
chd_stage       = pars['chdh']
chd_line        = [tuple(xy) for xy in river]

#%% Buildng Grid

g.build()
disv_props  = g.get_gridprops_vertexgrid()
vgrid       = flopy.discretization.VertexGrid(**disv_props)
idom        = np.ones([vgrid.nlay, vgrid.ncpl])
strt        = np.zeros([vgrid.nlay, vgrid.ncpl])+20
ixs         = flopy.utils.GridIntersect(vgrid, method = "vertex")

# TODO: RUN STEADYSTATE MODEL TO OBTAIN STARTING HEADS

#%% Loading reference fields
k_ref = pars['k_ref']
r_ref = pars['r_ref']
rivh  = pars['rivh']
sfac  = pars['sfac']

k_ref = np.flip(k_ref, axis  = 0)
r_ref = np.flip(r_ref, axis  = 0)
# logK = np.flip(np.flip(logK, axis = 0), axis = 1)

#%% Intersecting model grid with model features

rch_cells       = np.arange(vgrid.ncpl)
rch_lay         = np.zeros(vgrid.ncpl, dtype = int)
rch_cell2d      = list(zip(rch_lay,rch_cells))
rch_list        = list(zip(rch_cell2d, abs(r_ref.flatten())*sfac[0]))
for i in range(vgrid.ncpl):
    rch_list[i] = list(rch_list[i])

### Wells
result      = ixs.intersect(MultiPoint(welxy))
well_list   = []
for i, index in zip(result.cellids, range(len(result.cellids))):
    pump    = welq[index].astype(float)
    layer   = welay[index].astype(int)
    well_list.append([(layer,i),-pump])
    
### River
riverLS     = LineString(river)
l           = riverLS.length
riv_list    = []

for i in range(len(river)-1):
    rivl    = LineString(np.array([river[i],river[i+1]]))
    result  = ixs.intersect(rivl)
    for cell in result.cellids:
        xc,yc = vgrid.xyzcellcenters[0][cell],vgrid.xyzcellcenters[1][cell]
        riv_list.append([(0, cell), river_stages[0], rivC , river_stages[0]])
        
### Chd
chdLS       = LineString(chdl)
lchd        = chdLS.length
chd_list    = []

for i in range(len(chdl)-1):
    chdls   = LineString(np.array([chdl[i],chdl[i+1]]))
    result  = ixs.intersect(chdls)
    for cell in result.cellids:
        xc,yc = vgrid.xyzcellcenters[0][cell],vgrid.xyzcellcenters[1][cell]
        chd_list.append([(0, cell), chd_stage])

#%% Flopy Model definiiton

# simulation object
sim     = flopy.mf6.MFSimulation(sim_name           = sname,
                                 sim_ws             = sim_ws,
                                 verbosity_level    = 2)
# groundwater flow / model object
gwf     = flopy.mf6.ModflowGwf(sim,
                               modelname            = mname,
                               save_flows           = True)
# disv package
disv    = flopy.mf6.ModflowGwfdisv(model            = gwf,
                                   length_units     = "METERS",
                                   pname            = "disv",
                                   xorigin          = 0,
                                   yorigin          = 0,
                                   angrot           = 0,
                                   nogrb            = False,
                                   nlay             = disv_props["nlay"], 
                                   ncpl             = disv_props["ncpl"],
                                   nvert            = len(disv_props["vertices"]), 
                                   top              = disv_props["top"],
                                   botm             = disv_props["botm"], 
                                   idomain          = idom, 
                                   cell2d           = disv_props["cell2d"], 
                                   vertices         = disv_props["vertices"])
disv.export("./model_files/disv_ref.shp")
# npf package
npf     = flopy.mf6.ModflowGwfnpf(model             = gwf,
                                  k                 = k_ref)
# tdis package
tdis    = flopy.mf6.ModflowTdis(sim,
                                time_units          = "SECONDS",
                                perioddata          = [[60*60*6, 1, 1.0]])
# ims package
ims = flopy.mf6.ModflowIms(sim,
                           print_option             = "SUMMARY",
                           complexity               = "COMPLEX",
                           linear_acceleration      = "BICGSTAB")
# ic package
ic = flopy.mf6.ModflowGwfic(gwf, 
                            strt                    = strt)
sto = flopy.mf6.ModflowGwfsto(gwf, 
                              pname                 = "sto",
                              save_flows            = True,
                              iconvert              = 1,
                              ss                    = pars['ss'],
                              sy                    = pars['sy'],
                              steady_state          = {0: True},)
                               # transient             = {0: True},)
# rch package
rch = flopy.mf6.ModflowGwfrch(gwf,
                              stress_period_data    = {0:rch_list})
# wel package
wel = flopy.mf6.ModflowGwfwel(gwf,
                              stress_period_data    = {0:well_list})
# riv package
riv = flopy.mf6.ModflowGwfriv(gwf,
                              stress_period_data    = {0:riv_list})
# chd package
chd = flopy.mf6.ModflowGwfchd(gwf,
                              stress_period_data    = {0:chd_list})
# oc package
headfile            = "{}.hds".format(mname)
head_filerecord     = [headfile]
budgetfile          = "{}.cbb".format(mname)
budget_filerecord   = [budgetfile]
saverecord          = [("HEAD", "ALL"), ("BUDGET", "ALL")]
printrecord         = [("HEAD", "LAST")]
oc = flopy.mf6.ModflowGwfoc(gwf,
                            saverecord              = saverecord,
                            head_filerecord         = head_filerecord,
                            budget_filerecord       = budget_filerecord,
                            printrecord             = printrecord)

sim.write_simulation()
sim.run_simulation()

#%% Set steady-state solution as initial condition
ic.strt             = gwf.output.head().get_data()
sim.write_simulation()

#%% Plotting the necessary fields for comparison

# plot(gwf, ['logK', 'rch'])
plot(gwf, ['logK', 'rch', 'h'])
# plot(gwf, ['logK','h'], bc=False)
