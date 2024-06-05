import sys
sys.path.append('..')
# from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
import numpy as np
pars        = get()
n_mem       = pars['n_mem']
nprocs      = pars['nprocs']
clx         = pars['lx']
# sim, gwf = load_template_model(pars)

n_target = 10000
res = np.zeros((n_target,6))

for i in range(n_target):
    lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]*2),
                    np.random.randint(pars['dx'][1], clx[0][1]*2)])
    ang = np.random.uniform(0, np.pi)
    sigma = np.random.uniform(0.5, 3)
    if lx[0] < lx[1]:
        lx = np.flip(lx)
        if ang > np.pi/2:
            ang -= np.pi/2
        else:
            ang += np.pi/2
    elif lx[0] == lx[1]:
        lx[0] += 1
    
    res[i,0:3] = lx[0], lx[1], ang
    
    D = pars['rotmat'](ang)
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
    eigenvalues, eigenvectors =  np.linalg.eig(M)
    # res[i,3:] = extract_truth(eigenvalues, eigenvectors)
    res[i,3:] = pars['mat2cv'](eigenvalues, eigenvectors)

    
difference = res[:,0:3] - res[:,3:]
# difference[:,2] = difference[:,2]%np.pi
print(np.max(np.abs(difference)))