import sys
sys.path.append('..')
import numpy as np
from dependencies.load_template_model import load_template_model
from dependencies.model_params import get

def extract_truth(eigenvalues, eigenvectors):
    
    lxmat = 1/np.sqrt(eigenvalues)
    
    if lxmat[0] < lxmat[1]:
        lxmat = np.flip(lxmat)
        eigenvectors = np.flip(eigenvectors, axis = 1)
    
    if eigenvectors[0,0] > 0:
        ang = np.pi/2 -np.arccos(np.dot(eigenvectors[:,0],np.array([0,1])))    

    else:
        if eigenvectors[1,0] > 0:
            ang = np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))

        else:
            ang = np.pi -np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))

    return lxmat[0], lxmat[1], ang


pars        = get()
n_mem       = pars['n_mem']
nprocs      = pars['nprocs']
clx         = pars['lx']
sim, gwf = load_template_model(pars)

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
    
    eigenvalues, eigenvectors = eigenvalues, eigenvectors = np.linalg.eig(M)
    res[i,3:] = extract_truth(eigenvalues, eigenvectors)
    
difference = res[:,0:3] - res[:,3:]
print(np.max(np.abs(difference)))