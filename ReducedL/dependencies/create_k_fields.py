import gstools as gs
from gstools import krige
import numpy as np

def create_k_fields(gwf, pars: dict, pp_xy, pp_cid: np.ndarray, covtype = 'random', valtype = 'good'):
    dim = 2
    cov = pars['cov']
    clx     = pars['lx']
    angles  = pars['ang']
    sigma   = pars['sigma'][0]
    mu      = pars['mu'][0]
    cov     = pars['cov']
    k_ref   = pars['k_ref']

    mg = gwf.modelgrid
    xyz = mg.xyzcellcenters
    
    if covtype == 'random':
        lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]),
                       np.random.randint(pars['dx'][1], clx[0][1])])
        ang = np.random.uniform(0, np.pi)
        sigma = np.random.uniform(1, 5)
        if lx[0] < lx[1]:
            lx = np.flip(lx)
            if ang > 0:
                ang -= np.pi/2
            else:
                ang += np.pi/2
    elif covtype == 'good':
        lx = clx[0]
        ang = np.deg2rad(angles[0])
    # elif covtype == 'test':
    #     lx = np.array([np.random.randint(pars['dx'][0], clx[0][0]),
    #                    np.random.randint(pars['dx'][1], clx[0][1])])
    #     ang = np.deg2rad(10) + np.random.randn()/20
    
    if cov == 'Matern':
        model = gs.Matern(dim=dim, var=sigma, angles = ang, len_scale=lx)
    elif cov == 'Exponential':
        model = gs.Exponential(dim = dim, var = sigma, len_scale=lx, angles = ang)
    elif cov == 'Gaussian':
        model = gs.Gaussian(dim = dim, var = sigma, len_scale=lx, angles = ang)
    
    # starting k values at pilot points
    if valtype == 'good':
        pp_k = k_ref[pp_cid.astype(int)]
    elif valtype == 'random':
        pp_k = np.exp(np.random.randn(len(pp_cid)) + mu)
    
    krig = krige.Ordinary(model, cond_pos=(pp_xy[:,0], pp_xy[:,1]), cond_val = np.log(pp_k))
    field = krig((xyz[0], xyz[1]))

    # convert it to matrix form
    # rotation matrix 
    D = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) 
    M = D @ np.array([[1/lx[0]**2, 0],[0, 1/lx[1]**2]]) @ D.T
    
    # check angles --- so sollte es immer hinhauen, wenn wir zuerst sortieren
    eigenvalues, eigenvectors = np.linalg.eig(M) 
    # checken, in welchem quadranten wir sind. Da wir mod pi rechnen, ist die 
    # Orientierung immer in den oberen Quadranten, wenn gegen den Uhrzeigersinn
    # von der X-Achse aus gehend gedreht wird --> Die x-Komponente allein entscheidet,
    # in welchem Quadranten wir uns befinden
    if eigenvectors[0,0] > 0:
        # Wir sind im rechten Quadranten, klapt bei 140°, 170°, ab 135°
        angle = np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))
    else:
        # Wir sind im linken Quadranten
        # Funktioniert bei 40°. 20°, bis 45°
        angle1 = np.pi -np.arccos(np.dot(eigenvectors[:,0],np.array([1,0])))  
    
    tetst = True
    # Bei 125°, 89°, 50°, 90° ist die untere Formel korrekt
    # angle3 = np.pi -np.arccos(np.dot(eigenvectors[:,0],np.array([0,1])))
    
    # angle = np.pi -np.arccos(np.dot(eigenvectors[:,1],np.array([0,1])))
    # angle = np.pi -np.arccos(np.dot(eigenvectors[:,1],np.array([1,0])))
    
    # INSIGHT: Wir werden die Ellipsen nun immer sortieren mit der längeren l
    # am anfang  und mod pi, sodass wir immer nach oben orientiert sind. 
    # Mit arccos scheint es, als ob es zwischen 0 und 45 Grad, 45 und 135 Grad,
    # sowie 135 und 180 Grad unterschiedliche Formeln benötigt werden.
    
    # 1. Idee: Letzten Winkel anschauen und entsprechd nahe Formel nehmen. Wenn
    # es knapp wird, dann einfach zwei ausrechnen und die nähere nehmen.
    
    # 2. Idee: Nicht die Funktion von np nehmen, sondern Eigenvektoren selber
    # berechnen und schauen, ob es da eine Consistency gibt. Mir scheint, als ob
    # die Eigenvektoren die Plätze tauschen, je nachdem in welche Richtung sie 
    # gedreht sind. Viel Spaß heute Janek!!
    
    return np.exp(field[0]),  model, [M[0,0], M[1,0], M[1,1]], [lx[0], lx[1], ang]
    
        


