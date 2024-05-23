import sys
sys.path.append('..')
from dependencies.model_params import get
from Virtual_Reality.ReferenceModel import run_reference_model
from Virtual_Reality.Field_Generation import generate_fields
from dependencies.plotting.plot_fields import plot_fields
from dependencies.load_template_model import load_template_model
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm

pars        = get()
n_mem       = pars['n_mem']
nprocs      = pars['nprocs']

Kflat, Rflat, K, R = generate_fields(pars)
logK = np.log(K)

sim, gwf = load_template_model(pars)
plot_fields(gwf, pars,  np.log(Kflat), Rflat)

dx = pars['dx']
x = np.arange(dx[0]/2, 5000, dx[0])
y = np.arange(dx[1]/2, 2500, dx[1])


X, Y = np.meshgrid(x,y)

fig, axes   = plt.subplots(nrows=2, ncols=1, figsize=(8,6), sharex=True)
ax1, ax2 = axes
ax1.pcolor(X, Y, np.log(K), shading='auto', cmap=cm.bilbao_r)
ax2.pcolor(X, Y, R, shading='auto', cmap=cm.turku_r)

print(pars['ang']%180)





