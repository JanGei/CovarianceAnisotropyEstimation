# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:32:00 2024

@author: Anwender
"""

import shutil
import os
from functions.model_params import get
import flopy
import numpy as np
from functions.plot import movie

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
sfac            = pars['sfac']
river_stages    = pars['rivh']
mname           = pars['mname']
sname           = pars['sname']


sim = flopy.mf6.MFSimulation.load(sname,
                                  sim_ws = model_dir,
                                  verbosity_level = 1) 


#%% set transient forcing
gwf             = sim.get_model(mname)
rch             = gwf.rch
riv             = gwf.riv
sto             = gwf.sto
tdis            = sim.tdis

# # get stressperioddata
rch_spd         = rch.stress_period_data.get_data()
riv_spd         = riv.stress_period_data.get_data()

rch_cel         = rch_spd[0]['cellid']
riv_cel         = riv_spd[0]['cellid']
rivC            = riv_spd[0]['cond']

sto_tra         = {}

riv_list        = []
rch_list        = []
rivhl           = np.ones(np.shape(riv_cel))
perioddata =    []
# building tuples
for i in range(len(sfac)):
    rch_list.append(list(zip(rch_cel, abs(r_ref.flatten())*sfac[i])))
    riv_list.append(list(zip((0, riv_cel), rivhl*rivh[i], rivC , rivhl*rivh[i])))
    sto_tra[i] = True
    perioddata.append([60*60*6, 1, 1.0])

# set new data 
tdis.perioddata.set_data(perioddata)
tdis.nper.set_data(len(perioddata))
sto.transient.set_data(sto_tra)
rch.stress_period_data.set_data(rch_spd)
riv.stress_period_data.set_data(riv_spd)

sim.write_simulation()
sim.run_simulation()

# head = gwf.output.head().get_data()
heads = np.empty((len(perioddata), 1, gwf.disv.ncpl.data))
for  i in range(len(perioddata)):
    heads[i,0,:] = flopy.utils.binaryfile.HeadFile(model_dir+mname+'.hds').get_data(kstpkper=(0, i))
    
np.save('model_data/head_ref.npy', heads)

movie(gwf)
    

        
    
    
    
    
    