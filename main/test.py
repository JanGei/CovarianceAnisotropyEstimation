# from dependencies.randomK import randomK2, randomK
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.interpolate import griddata
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.plot import plot_k_fields
from dependencies.randomK_points import randomK_points

pars = get()
sim, gwf = load_template_model(pars)
mg = gwf.modelgrid
xyz = gwf.modelgrid.xyzcellcenters
xmin, xmax, ymin, ymax = mg.extent
dxmin = np.min([max(sublist) - min(sublist) for sublist in mg.xvertices])
dymin = np.min([max(sublist) - min(sublist) for sublist in mg.yvertices])
dx = [dxmin, dymin]

cxy = np.vstack((xyz[0], xyz[1])).T

ang = 90
Ctype = 'Matern'
sigY = 3
lx = np.array([1500, 200])
Kg = 2e-4

field, K = randomK_points(mg.extent, cxy, dx,lx,ang,sigY,Ctype,Kg)

plot_k_fields(gwf, pars,[field, K.flatten()])
# corval, K2 = randomK2(coordinates, dx, lx, ang, sigY, Ctype, Kg)
