import sys
sys.path.append('..')
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from objects.MFModel import MFModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dependencies.intersect_with_grid import intersect_with_grid


pars = get()
# copy template model to ensemble folder
model_dir = create_Ensemble(pars)
sim, gwf = load_template_model(pars)
obs_cid = intersect_with_grid(gwf, pars['obsxy'])
ang = np.pi/3.2
lx = np.array([420, 169])
l_angs = np.array([lx[0], lx[1], ang])
D = pars['rotmat'](-ang)
M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T

M1 = MFModel(model_dir[0][0], pars, obs_cid, l_angs, [M[0,0], M[0,1],M[1,1]])



n1 = np.arange(M1.threshhold, 5000, 10)
n2 = np.arange(0, M1.threshhold, 10)
inter = np.empty(np.shape(n1))

for j in range(len(n1)):
    inter[j] = M1.reduce_corL(n1[j])
    print(f'{n1[j]} becomes {inter[j]}')
    
plt.figure()
plt.plot(n1, inter)
plt.plot(n2, n2)
# plt.plot(res[:,3], res[:,1])
