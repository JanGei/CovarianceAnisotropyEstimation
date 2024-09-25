import sys
sys.path.append('..')
from dependencies.model_params import get
from Virtual_Reality.Field_Generation import generate_fields
from Virtual_Reality.ReferenceModel import create_reference_model
from dependencies.plotting.plot_fields import plot_fields
from dependencies.load_template_model import load_template_model
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

pars        = get()
n_mem       = pars['n_mem']
nprocs      = pars['nprocs']

generate_fields(pars)
logK = np.log(np.loadtxt(pars['k_r_d'], delimiter = ','))
R = np.loadtxt(pars['r_r_d'], delimiter = ',')

sim, gwf = load_template_model(pars)
plot_fields(gwf, pars,  logK, R)

dx = pars['dx']
x = np.arange(dx[0]/2, 5000, dx[0])
y = np.arange(dx[1]/2, 2500, dx[1])


X, Y = np.meshgrid(x,y)

fig, axes   = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
ax1, ax2 = axes
ax1.pcolor(X, Y, np.reshape(logK, (50,100), order = 'A'), shading='auto', cmap=cm.bilbao_r)
ax2.pcolor(X, Y, np.reshape(R, (50,100), order = 'A'), shading='auto', cmap=cm.turku_r)

print(pars['ang']%180)





