import sys
sys.path.append('..')
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
from dependencies.copy import create_Ensemble
from objects.MFModel import MFModel
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


pars = get()
# copy template model to ensemble folder
model_dir = create_Ensemble(pars)
sim, gwf = load_template_model(pars)

ang = np.pi/3.2
lx = np.array([420, 169])
l_angs = np.array([lx[0], lx[1], ang])
D = pars['rotmat'](-ang)
M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T

M1 = MFModel(model_dir[0], pars, 500, l_angs, [M[0,0], M[0,1],M[1,1]])



n1 = 200
n2 = 100
res = np.zeros((n1,4))

for j in range(n1):
    inter = a = M1.check_corrL(10*j,7.5*j,np.pi/3)
    res[j,0] = inter[0]
    res[j,1] = inter[1]
    res[j,2] = 10*j
    res[j,3] = 7.5*j
    
plt.figure()
plt.plot(res[:,2], res[:,0])
# plt.plot(res[:,3], res[:,1])
