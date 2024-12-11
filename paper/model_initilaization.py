import sys
sys.path.append('..')
sys.path.append('../main')

import numpy as np
import flopy
from cmcrameri import cm
import matplotlib.patches as patches
from dependencies.randomK import randomK
from dependencies.model_params import get
from functions.kriging import kriging
from functions.conditional_k import conditional_k
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from flopy.utils.gridintersect import GridIntersect
from shapely.geometry import LineString,MultiPoint
import os
from flopy.discretization.structuredgrid import StructuredGrid
from flopy.utils.gridgen import Gridgen
import shutil
from Virtual_Reality.Field_Generation import generate_fields
import sys

cwd = os.getcwd()
pars = get()

#%% Model Parameters
nx      = pars['nx']
dx      = pars['dx']
toph    = pars['top']
nlay    = pars['nlay'][0]
mname   = pars['mname']
sname   = pars['sname']
sim_ws  = "model_files/"
gg_ws   = "model_files/"

#%% Grid Generation
Lx = nx[0] * dx[0]
Ly = nx[1] * dx[1]

delr = np.ones(nx[0])*Lx/nx[0]
delc = np.ones(nx[1])*Ly/nx[1]

top     =  np.array([np.ones((nx[1],nx[0]))]*toph)
botm    =  np.array([np.zeros((nx[1],nx[0]))])

strgrd = StructuredGrid(delc=delc.astype(int), delr=delr.astype(int), top=top, botm=botm, nlay=nlay)
if os.path.exists(gg_ws):
    shutil.rmtree(gg_ws)
    os.mkdir(gg_ws)
g = Gridgen(strgrd, model_ws=gg_ws)


#%% Northern Boudnary - Fixed head
chdl_west = np.array([[0,0],[5000,0]])
chdl_east = np.array([[0,2500],[5000,2500]])
chd_stage_west = 13
chd_stage_east = 15

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
                              steady_state          = {0: True},
                              transient             = {0: False})

sim.write_simulation()
#%% Generating and loading reference fields
generate_fields(pars)
print('Fields are generated')
k_ref = np.loadtxt(pars['k_r_d'], delimiter = ',')

### Chd
# chdLS       = LineString(chdl)
# lchd        = chdLS.length
chd_list    = []

for i in range(len(chdl_west)-1):
    chdls_west = LineString(np.array([chdl_west[i],chdl_west[i+1]]))
    chdls_east = LineString(np.array([chdl_east[i],chdl_east[i+1]]))
    result_west = ixs.intersect(chdls_west)
    result_east = ixs.intersect(chdls_east)
    for cell in result_west.cellids:
        chd_list.append([(0, cell), chd_stage_west])
    for cell in result_east.cellids:
        chd_list.append([(0, cell), chd_stage_east])


rch_list = []
for i in range(vgrid.ncpl):
    rch_list.append([(0, i), 1e-8])
    
# npf package
npf     = flopy.mf6.ModflowGwfnpf(model             = gwf,
                                  k                 = k_ref)


# ic package
ic = flopy.mf6.ModflowGwfic(gwf, 
                            strt                    = strt)

# rch package
rch = flopy.mf6.ModflowGwfrch(gwf,
                              stress_period_data    = {0:rch_list})

# chd package
chd = flopy.mf6.ModflowGwfchd(gwf,
                              stress_period_data    = {0:chd_list})


#%% Set steady-state solution as initial condition
sim.write_simulation()
sim.run_simulation()
ic.strt.set_data(gwf.output.head().get_data())
ic.write()


#%%

sigma = pars['sigma'][0]
cxy = np.vstack([gwf.modelgrid.xcellcenters, gwf.modelgrid.ycellcenters]).T

x = np.arange(pars['dx'][0]/2, pars['nx'][0]*pars['dx'][0], pars['dx'][0])
y = np.arange(pars['dx'][1]/2, pars['nx'][1]*pars['dx'][1], pars['dx'][1])

# Grid in Physical Coordinates
X, Y = np.meshgrid(x, y)

rch = gwf.rch

K = randomK(np.deg2rad(pars['ang'][0]), pars['sigma'][0], pars['cov'], pars, ftype = 'K', random = False)
k_true =  griddata((X.ravel(order = 'F'), Y.ravel(order = 'F')), K.ravel(order = 'F'),
                 (cxy[:,0], cxy[:,1]), method='nearest')
k_true_grid = np.reshape(k_true, (50,100), order = 'C')
#%%

obs_cellid = np.array([[42, 12],
                       [14, 21],
                       [31, 33],
                       [10, 50],
                       [44, 73],
                       [19, 74],
                       [28, 95]
                       ])
obs_xy = np.zeros(obs_cellid.shape)
for i in range(len(obs_xy)):
    obs_xy[i,0] = obs_cellid[i,1] * 50
    obs_xy[i,1] = 2500 - obs_cellid[i,0] * 50

ixs = GridIntersect(gwf.modelgrid)
result = ixs.intersect(MultiPoint(obs_xy))
pp_k = [[],[],[]]
pp_k1 = [[],[],[]]
for i, index in zip(result.cellids, range(len(result.cellids))):
    pp_k[0].append(np.log(k_true_grid[obs_cellid[index,0],obs_cellid[index,1]]))
    pp_k[1].append(np.log(k_true_grid[obs_cellid[index,0],obs_cellid[index,1]])* np.random.uniform(0.9,1.1))
    pp_k[2].append(np.random.normal(pars['mu'][0], pars['sigma'][0]))


# print(pp_k[1][[5]])
lx = [pars['lx'][0], np.array([1600,400])]
ang = [np.deg2rad(pars['ang'][0]), np.deg2rad(195)]
conditional = False
if conditional:
    v_a_c1, f_g1 = conditional_k(cxy, dx, lx[0], ang[0], sigma, pars, np.array(pp_k[1]), obs_xy)
    v_a_c2, f_g2 = conditional_k(cxy, dx, lx[1], ang[1], sigma, pars, np.array(pp_k[1]), obs_xy)
    v_a_c3, f_g3 = conditional_k(cxy, dx, lx[1], ang[1], sigma, pars, np.array(pp_k[2]), obs_xy)
else:
    v_a_c1, f_g1 = kriging(cxy, dx, lx[0], ang[0], sigma, pars, pp_k[1], obs_xy)
    v_a_c2, f_g2 = kriging(cxy, dx, lx[1], ang[1], sigma, pars, pp_k[1], obs_xy)
    v_a_c3, f_g3 = kriging(cxy, dx, lx[1], ang[1], sigma, pars, pp_k[2], obs_xy)



fields = [k_true, v_a_c1, v_a_c2, v_a_c3]
lengths = [lx[0], lx[0], lx[1], lx[1]]
angles = [ang[0], ang[0], ang[1], ang[1]]

hfields = []
for i, field in enumerate(fields):
    gwf.npf.k.set_data([field])
    gwf.sto.transient.set_data({0: False})
    gwf.sto.steady_state.set_data({0: True})
    sim.write_simulation()
    sim.run_simulation()
    hfields.append(gwf.output.head().get_data())
    if i == -1:
        gwf.npf.k.set_data([k_true])
        gwf.npf.write()

#%% plot        
nrows, ncols = 3,3
fig, axes = plt.subplots(nrows = nrows, ncols = ncols,
                         sharex = True, sharey = True,
                         figsize=(12,6.25), constrained_layout=True)

ax0 = flopy.plot.PlotMapView(model=gwf, ax=axes[0,0])
c0 = ax0.plot_array(np.log(k_true), cmap=cm.bilbao_r, alpha=1)
mink = np.log(np.min(k_true))
maxk = np.log(np.max(k_true))
arc_radius = 300
levels = np.linspace(13.2, 16.2, 7)
for i in range(ncols):

    # Plot data for the first row (axes[0, i])
    ax1 = flopy.plot.PlotMapView(model=gwf, ax=axes[0, i])
    c1 = ax1.plot_array(np.log(fields[i])/np.log(10), cmap=cm.bilbao_r, alpha=1, vmin=mink/np.log(10), vmax=maxk/np.log(10))
    axes[0, i].set_aspect('equal')
    axes[0, i].scatter(obs_xy[:, 0], obs_xy[:, 1], color='black', s=10)

    # Plot data for the last row (axes[2, i])
    ax = flopy.plot.PlotMapView(model=gwf, ax=axes[2, i])
    c2 = ax.plot_array(hfields[i], cmap=cm.devon_r, alpha=1)
    contour = axes[2, i].contour(X,
                                 Y,
                                 np.flip(np.reshape(hfields[i], X.shape), axis=0),
                                 levels=levels,
                                 vmin=np.min(hfields[i]), 
                                 vmax=np.max(hfields[i]), 
                                 cmap='gray')
    labels = axes[2, i].clabel(contour, inline=True, fontsize=10, fmt='%1.1f', colors='black', inline_spacing=5)
    for label in labels:
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.5))
    axes[2, i].set_aspect('equal')

    for j in range(nrows):
        if j != 2:
            axes[j, i].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            axes[j, i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        else:
            axes[j, i].set_xticks([1000, 2500, 4000])
            axes[j, i].set_xticklabels([1, 2.5, 4], fontsize = 12)
            axes[j, i].set_xlabel('Easting (km)', fontsize = 12)
            if i == 0:
                axes[j, 0].set_yticks([1000, 2000])
                axes[j, 0].set_yticklabels([1, 2], fontsize = 12)
                axes[j, 0].set_ylabel('Northing (km)', fontsize = 12)

    # Add ellipses to the second row (axes[1, i])
    ellipse = patches.Ellipse((2500, 1250),
                              lengths[i][0]*2,
                              lengths[i][1]*2,
                              angle=np.rad2deg(angles[i]),
                              fill=False,
                              color='black',
                              alpha=0.5,
                              zorder=1,
                              linewidth=3)
    axes[1, i].add_patch(ellipse)
    axes[1, i].set_aspect('equal')
    
    x_center, y_center = (2500,1250)
    x_length = 3000 # Length of x-axis
    y_length = 2300  # Length of y-axis

    # Plot x-axis (aligned with ellipse's major axis)
    x_axis_srt = [x_center - x_length/2, 
                  y_center ]
    x_axis_end = [x_center + x_length/2, 
                  y_center]
    
    axes[1, i].plot([x_axis_srt[0], x_axis_end[0]], [y_center, x_axis_end[1]], 
                    color='black', lw=1, zorder=0)

    # Plot y-axis (aligned with ellipse's minor axis)
    y_axis_srt = [x_center, 
                  y_center - y_length/2]
    y_axis_end = [x_center, 
                  y_center +  y_length/2]

    axes[1, i].plot([y_axis_srt[0], y_axis_end[0]], [y_axis_srt[1], y_axis_end[1]], 
                    color='black', lw=1, zorder=0)
    
    # plot major/minor axis
    x_center, y_center = (2500,1250)
    x_length = lengths[i][0]  # Length of x-axis
    y_length = lengths[i][1]  # Length of y-axis

    # Plot x-axis (aligned with ellipse's major axis)
    x_axis_end = [x_center + x_length * np.cos(angles[i]), 
                  y_center + x_length * np.sin(angles[i])]
    axes[1, i].plot([x_center, x_axis_end[0]], [y_center, x_axis_end[1]], 
                    color='red', lw=2, zorder=2)

    # Plot y-axis (aligned with ellipse's minor axis)
    y_axis_end = [x_center - y_length * np.sin(angles[i]), 
                  y_center + y_length * np.cos(angles[i])]
    axes[1, i].plot([x_center, y_axis_end[0]], [y_center, y_axis_end[1]], 
                    color='blue', lw=2, zorder=2)
    
    angle_arc = patches.Arc((2500,1250), 
                            arc_radius*2, arc_radius*2, 
                            theta1=0, theta2=np.rad2deg(angles[i]),  # Start at 0 degrees, end at the ellipse angle
                            color='green', lw=2, zorder=3)
    axes[1, i].add_patch(angle_arc)
    
    # Adding text
    if i == 2:
        y1c = -450
        x1c = -750
        y2c = -300
        ha1 = 'left'
        ha2 = 'left'
    else:
        y1c = -250
        x1c = -250
        y2c = -250
        ha1 = 'right'
        ha2 = 'right'
    axes[1, i].text(x_axis_end[0]+x1c, x_axis_end[1]+y1c, r'Major Axis $\equiv l_1$', color='red', fontsize=12,
                    ha=ha1, va='bottom')
    axes[1, i].text(y_axis_end[0], y_axis_end[1]+y2c, r'Minor Axis $\equiv l_2$', color='blue', fontsize=12,
                    ha=ha2, va='bottom')
    axes[1, i].text(2750, 1450, r'$\alpha$', color='green', fontsize=12, fontweight = 'bold',
                    ha='left', va='bottom')

contour_plots = [axes[2, i].contour(X, Y, np.flip(np.reshape(hfields[i], X.shape), axis=0), levels=levels, cmap='gray') for i in range(ncols)]        
# cbar2 = fig.colorbar(contour_plots[0].collections[0], ax=axes[2, :], location='right', pad=0.01, shrink=0.8, aspect=10)
cbar0 = fig.colorbar(c1, ax=axes[0, :], location='right', pad=0.01, shrink=0.8, aspect=10)
cbar2 = fig.colorbar(c2, ax=axes[2, :], location='right', pad=0.01, shrink=0.8, aspect=10)
# cax1 = divider1.append_axes("right", size="5%", pad=pad)
# cbar1 = fig.colorbar(r, cax=cax1)
cbar0.set_label(r'Conductivity ($log_{10}$(m/s))', fontsize=12)
cbar2.set_label('Head (m)', fontsize=12)
axes[0,0].set_title('Reference', fontsize = 14, fontweight = 'bold')   
axes[0,1].set_title('Correct covariance function', fontsize = 14, fontweight = 'bold') 
axes[0,2].set_title('Wrong covariance function', fontsize = 14, fontweight = 'bold') 

# fig.tight_layout()
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.001, wspace=0.3)        
if conditional:
    fig.savefig(os.path.join(cwd, 'plots', 'fig_Covariance_Function_Compare_Conditional.png'), transparent=True, dpi=300)
else:
    fig.savefig(os.path.join(cwd, 'plots', 'fig_Covariance_Function_Compare.png'), transparent=True, dpi=300)
# divider0 = make_axes_locatable(axes[0])
# cax0 = divider0.append_axes("right", size="5%", pad=pad)  # Adjust size and pad for better spacing
# cbar0 = fig.colorbar(c0, cax=cax0)
# cbar0.mappable.set_clim(kmin, kmax)
# cbar0.set_label('Log-Conductivity (log(m/s))', fontsize=fontsize)
# cbar0.ax.tick_params(labelsize=fontsize)
# axes[0].set_aspect('equal')  # Change to 'auto' to prevent squishing
# axes[0].set_ylabel('Y-axis', fontsize=fontsize)
# axes[0].tick_params(axis='both', which='major', labelsize=fontsize)