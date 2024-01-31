# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:32:00 2024

@author: Anwender
"""

import shutil
import os
from functions.model_params import get
from shapely.geometry import MultiPoint
import flopy
import numpy as np
from functions.plot import movie
import json

def copy_model(orig_dir, model_dir):
    # Check if the destination folder already exists
    if os.path.exists(model_dir):
        # Remove the existing destination folder and its contents
        shutil.rmtree(model_dir)

    # Copy the model folder to new folder
    shutil.copytree(orig_dir, model_dir)

pars            = get()

orig_dir    = pars['sim_ws']
mname       = pars['mname']
sname       = pars['sname']
model_dir   = 'transient_model/'
copy_model(orig_dir, model_dir)

#%% load everything

r_ref           = pars['r_ref']
rivh            = pars['rivh']
rivd            = pars['rivd']
sfac            = pars['sfac']
river_stages    = pars['rivh']
mname           = pars['mname']
sname           = pars['sname']
welq            = pars['welq'] 
welst           = pars['welst'] 
welnd           = pars['welnd'] 
welay           = pars['welay'] 
welxy           = pars['welxy'] 
obsxy           = pars['obsxy'] 


sim = flopy.mf6.MFSimulation.load(sname,
                                  sim_ws = model_dir,
                                  verbosity_level = 1) 


#%% set transient forcing
gwf             = sim.get_model(mname)
rch             = gwf.rch
riv             = gwf.riv
sto             = gwf.sto
tdis            = sim.tdis
wel             = gwf.wel
ic              = gwf.ic

# # get stressperioddata
rch_spd         = rch.stress_period_data.get_data()
riv_spd         = riv.stress_period_data.get_data()
wel_spd         = wel.stress_period_data.get_data()

rch_cel         = rch_spd[0]['cellid']
riv_cel         = riv_spd[0]['cellid']
rivC            = riv_spd[0]['cond']

sto_tra         = {}

wel_list        = {}
riv_list        = {}
rch_list        = {}
rivhl           = np.ones(np.shape(riv_cel))
perioddata      = []
# building tuples
# REBUILD IN TERMS OF SPDS <-- CONTINUE HERE
time_d          = 0
for i in range(len(sfac)):
    # recharge
    rch_list[i] = rch_spd[0].copy()
    rch_list[i]['recharge'] = abs(np.array(r_ref).flatten()) * sfac[i]
    # river
    riv_list[i] = riv_spd[0].copy()
    riv_list[i]['stage'] = rivhl * rivh[i]
    # wel
    wel_list[i] = wel_spd[0].copy()
    for j in range(len(wel_list[i]['q'])):
        if welst[j] <= time_d and welnd[j] > time_d:
            wel_list[i]['q'][j] = -welq[j]
        else:
            wel_list[i]['q'][j] = 0
    # sto & tdis
    sto_tra[i] = True
    perioddata.append([60*60*6, 1, 1.0])
    
    time_d += 0.25

#%%
    
# set new data 
tdis.perioddata.set_data(perioddata)
tdis.nper.set_data(len(perioddata))
sto.transient.set_data(sto_tra)
rch.stress_period_data.set_data(rch_list)
riv.stress_period_data.set_data(riv_list)
wel.stress_period_data.set_data(wel_list)

sim.write_simulation()
sim.run_simulation()

#%% Set steady-state solution as initial condition
heads = np.empty((len(perioddata), 1, gwf.disv.ncpl.data))
for i in range(len(perioddata)):
    heads[i,0,:] = gwf.output.head().get_data(kstpkper=(0, i))
    
#%% Generate Observations
ixs         = flopy.utils.GridIntersect(gwf.modelgrid, method = "vertex")
result      = ixs.intersect(MultiPoint(obsxy))

obs         = {}
for i, cellid in enumerate(result.cellids):
    obs[i] = {'cellid':cellid,
              'h_obs':np.empty((1,len(perioddata)))}
    
for i in range(len(perioddata)):
    for j in range(len(obs)):
        obs[j]['h_obs'][0,i] = heads[i,0,obs[j]['cellid']]


#%%
np.save('model_data/head_ref.npy', heads)
np.save('model_data/obs_ref.npy', obs)
# find a way to store dict
# with open('model_data/obs.json', 'w') as json_file:
#     json.dump(obs, json_file)
# lb = flopy.utils.Mf6ListBudget(model_dir+'Reference.cbb')
# data = lb.get_dataframes()
# movie(gwf, diff = False, contour = True)
    

        
    
    
    
    
    