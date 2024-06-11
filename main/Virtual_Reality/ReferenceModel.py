# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:01:12 2023

@author: Janek
"""
import flopy
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.utils.gridgen import Gridgen
from shapely.geometry import LineString, MultiPoint
import numpy as np
from Virtual_Reality.Field_Generation import generate_fields
from dependencies.convert_transient import convert_to_transient
# from Virtual_Reality.transient_run import transient_run
from dependencies.plotting.plot_fields import plot_fields
import sys

def create_reference_model(pars):
    #%% Model Parameters
    nx      = pars['nx']
    dx      = pars['dx']
    toph    = pars['top']
    nlay    = pars['nlay'][0]
    mname   = pars['mname']
    sname   = pars['sname']
    sim_ws  = pars['sim_ws']
    gg_ws   = pars['gg_ws']
    
    #%% Grid Generation
    Lx = nx[0] * dx[0]
    Ly = nx[1] * dx[1]
    
    delr = np.ones(nx[0])*Lx/nx[0]
    delc = np.ones(nx[1])*Ly/nx[1]
    
    top     =  np.array([np.ones((nx[1],nx[0]))]*toph)
    botm    =  np.array([np.zeros((nx[1],nx[0]))])
    
    strgrd = StructuredGrid(delc=delc.astype(int), delr=delr.astype(int), top=top, botm=botm, nlay=nlay)
    
    g = Gridgen(strgrd, model_ws=gg_ws)
    
    #%% Well Location
    welxy   = pars['welxy']
    welq    = pars['welq']
    welay   = pars['welay']
    
    # possible refinements
    g.add_refinement_features(welxy, "point", 4, range(nlay))
    
    #%% Southern Boudnary - river
    river           = pars['river']
    rivd            = pars['rivd']
    river_stages    = np.genfromtxt(pars['rh_d'],delimiter = ',', names=True)['Wert']
    rivC            = pars['rivC']
    riv_line        = [tuple(xy) for xy in river]
    
    # possible refinements
    g.add_refinement_features([riv_line], "line", 3, range(nlay))
    
    #%% Northern Boudnary - Fixed head
    chdl            = pars['chd']
    chd_stage       = pars['chdh']
    # chd_line        = [tuple(xy) for xy in river]
    
    #%% Buildng Grid
    
    g.build()
    disv_props  = g.get_gridprops_vertexgrid()
    vgrid       = flopy.discretization.VertexGrid(**disv_props)
    idom        = np.ones([vgrid.nlay, vgrid.ncpl])
    strt        = np.zeros([vgrid.nlay, vgrid.ncpl])+20
    ixs         = flopy.utils.GridIntersect(vgrid, method = "vertex")
    
     
    #%% Flopy Model definiiton - Core packages
    
    # simulation object
    sim     = flopy.mf6.MFSimulation(sim_name           = sname,
                                     sim_ws             = sim_ws,
                                     verbosity_level    = 0)
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
    
    # tdis package
    tdis    = flopy.mf6.ModflowTdis(sim,
                                    time_units          = "SECONDS",
                                    perioddata          = [[60*60*6, 1, 1.0]])
    
    # ims package
    ims = flopy.mf6.ModflowIms(sim,
                               print_option             = "SUMMARY",
                               complexity               = "COMPLEX",
                               linear_acceleration      = "BICGSTAB")
    
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
    
    sto = flopy.mf6.ModflowGwfsto(gwf, 
                                  pname                 = "sto",
                                  save_flows            = True,
                                  iconvert              = 1,
                                  ss                    = pars['ss'],
                                  sy                    = pars['sy'],
                                  steady_state          = {0: True},)
                                   # transient             = {0: True},)

    sim.write_simulation()
    #%% Generating and loading reference fields
    generate_fields(pars)
    print('Fields are generated')
    k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')
    r_ref = np.loadtxt(pars['r_r_d'], delimiter = ',')
    sfac  = np.genfromtxt(pars['sf_d'],delimiter = ',', names=True)['Wert']
    
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
    # riverLS     = LineString(river)
    # l           = riverLS.length
    riv_list    = []
    
    for i in range(len(river)-1):
        rivl    = LineString(np.array([river[i],river[i+1]]))
        result  = ixs.intersect(rivl)
        for cell in result.cellids:
            # xc,yc = vgrid.xyzcellcenters[0][cell],vgrid.xyzcellcenters[1][cell]
            riv_list.append([(0, cell), river_stages[0], rivC , river_stages[0]-rivd])
            
    ### Chd
    # chdLS       = LineString(chdl)
    # lchd        = chdLS.length
    chd_list    = []
    
    for i in range(len(chdl)-1):
        chdls   = LineString(np.array([chdl[i],chdl[i+1]]))
        result  = ixs.intersect(chdls)
        for cell in result.cellids:
            # xc,yc = vgrid.xyzcellcenters[0][cell],vgrid.xyzcellcenters[1][cell]
            chd_list.append([(0, cell), chd_stage])
    
    # npf package
    npf     = flopy.mf6.ModflowGwfnpf(model             = gwf,
                                      k                 = k_ref)

    
    # ic package
    ic = flopy.mf6.ModflowGwfic(gwf, 
                                strt                    = strt)
    
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
    
    if pars['inspec'] and pars['setup'] == 'office':
        print(pars['mu'][0], np.mean(np.log(k_ref)))
        print(pars['mu'][1]/(86400), np.mean(r_ref))
        plot_fields(gwf, pars, np.log(k_ref), r_ref)
        sys.exit()
    
    #%% Set steady-state solution as initial condition
    sim.write_simulation()
    sim.run_simulation()
    ic.strt.set_data(gwf.output.head().get_data())
    ic.write()
    
    
    #%% Run transient simulation
    sim = convert_to_transient(pars['trs_ws'], pars)
    
    # transient_run(pars)
    
    # plot(gwf, ['logK', 'rch'])
    # plot(gwf, ['logK', 'rch', 'h'], bc = True)
    # plot(gwf, ['logK','h'], bc=False)

