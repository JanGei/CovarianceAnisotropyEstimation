# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:32:00 2024

@author: Anwender
"""

import shutil
import os
from functions.model_params import get
import flopy

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
model_dir   = 'transient_model'
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
                                  verbosity_level = 2) 


#%% set transient forcing
gwf             = sim.get_model(mname)
rch             = gwf.rch
riv             = gwf.riv
sto             = gwf.sto

# # get stressperioddata
rch_spd         = rch.stress_period_data.get_data()

rch_spd[0]['recharge'][0] / r_ref[0]
# for i in range(len(sfac)):
    