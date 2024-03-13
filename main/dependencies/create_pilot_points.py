from scipy.stats import qmc
import numpy as np
import flopy
from shapely.geometry import MultiPoint
from shapely.geometry import Point

    
def create_pilot_points(gwf, pars:dict,):

    nPP = pars['n_PP']
    mg = gwf.modelgrid
    ixs  = flopy.utils.GridIntersect(mg, method = "vertex")
    xyzex = mg.xyzextent
    xyz = mg.xyzcellcenters
    xy = list(zip(xyz[0], xyz[1]))
    
    welxy = pars['welxy']
    obsxy = pars['obsxy']
    
    
    welcid = ixs.intersect(MultiPoint(welxy))
    obscid = ixs.intersect(MultiPoint(obsxy))
     
    blocked_cid = np.concatenate((welcid.cellids,obscid.cellids))
    pp_cid_accepted = []
    pp_xy_accepted = []
    
    n_test = nPP
    sampler = qmc.Halton(2, scramble= False)
    
    while len(pp_cid_accepted) != nPP:
        
        pp_xy_proposal = sampler.random(n = n_test) * np.array([xyzex[1], xyzex[3]])
        
        pp_cid_proposal = np.zeros(len(pp_xy_proposal))
        for i, point in  enumerate(pp_xy_proposal):
            pp_cid_proposal[i] = ixs.intersect(Point(point)).cellids.astype(int)
        
        
        common_cells = np.intersect1d(pp_cid_proposal, blocked_cid)
        
        pp_cid_accepted = np.setdiff1d(pp_cid_proposal, common_cells)
        
        # get xy coordinated from proposed cells
        
        if len(pp_cid_accepted) == nPP:
            pp_xy_accepted = np.array([xy[int(i)] for i in pp_cid_accepted])
            

        
        if len(pp_cid_accepted) > nPP:
            n_test -= int((len(pp_cid_proposal) - nPP) /2)
        elif len(pp_cid_accepted) < nPP:
            n_test += int((nPP - len(pp_cid_proposal) ) /2)
            
            
    
    return pp_cid_accepted.astype(int), pp_xy_accepted.astype(int)











