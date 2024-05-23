from scipy.stats import qmc
import numpy as np
import flopy
from shapely.geometry import MultiPoint, Point, Polygon
import geopandas as gpd

    
def create_pilot_points(gwf, pars:dict,):

    nPP = pars['n_PP']
    mg = gwf.modelgrid
    ixs  = flopy.utils.GridIntersect(mg, method = "vertex")
    vert = mg.xyzvertices
    xmax = np.max([np.max(list) for list in vert[0]])
    ymax = np.max([np.max(list) for list in vert[1]])
    # xyzex = mg.xyzextent
    xyz = mg.xyzcellcenters
    xy = list(zip(xyz[0], xyz[1]))
    
    welxy = pars['welxy']
    obsxy = pars['obsxy']
    
    welcid = ixs.intersect(MultiPoint(welxy))
    obscid = ixs.intersect(MultiPoint(obsxy))
    
    # omit the firs two rows/columns of the model domain
    dx_x = np.max([max(xvertices) - min(xvertices) for xvertices in mg.xvertices])
    dx_y = np.max([max(yvertices) - min(yvertices) for yvertices in mg.yvertices])
    nc = pars["omitc"]
    
    blocked_cid = np.concatenate((welcid.cellids,obscid.cellids))
    pp_cid_accepted = []
    pp_xy_accepted = []
    
    n_test = nPP
    sampler = qmc.Halton(2, scramble = False)
    
    while len(pp_cid_accepted) != nPP:
        
        pp_xy_proposal = (sampler.random(n = n_test) * np.array([xmax-2*nc*dx_x, ymax-2*nc*dx_y])) + np.array([nc*dx_x, nc*dx_y])
        
        pp_cid_proposal = np.zeros(len(pp_xy_proposal))
        for i, point in  enumerate(pp_xy_proposal):
            pp_cid_proposal[i] = ixs.intersect(Point(point)).cellids.astype(int)[0]
        
        
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


def create_pilot_points_even(gwf, pars:dict,):

    nPP = pars['n_PP']
    mg = gwf.modelgrid
    ixs  = flopy.utils.GridIntersect(mg, method = "vertex")
    vert = mg.xyzvertices
    xmax = np.max([np.max(list) for list in vert[0]])
    ymax = np.max([np.max(list) for list in vert[1]])
    extent = mg.extent
    xyz = mg.xyzcellcenters
    xy = list(zip(xyz[0], xyz[1]))
    
    welxy = pars['welxy']
    obsxy = pars['obsxy']
    
    welcid = ixs.intersect(MultiPoint(welxy))
    obscid = ixs.intersect(MultiPoint(obsxy))
    
    # omit the firs two rows/columns of the model domain
    dx_x = np.max([max(xvertices) - min(xvertices) for xvertices in mg.xvertices])
    dx_y = np.max([max(yvertices) - min(yvertices) for yvertices in mg.yvertices])
    nc = pars["omitc"]

    
    blocked_cid = np.concatenate((welcid.cellids,obscid.cellids))
    pp_cid_accepted = []
    pp_xy_accepted = []
    
    ratio = int(extent[1] / extent[3])
    nPPy = int(np.sqrt(nPP/ratio))
    nPPx = int(ratio * nPPy)
            
    xratio = np.linspace(0, 1, nPPx+2)
    yratio = np.linspace(0, 1, nPPy+2)
    offset = 0
    
    while len(pp_cid_accepted) != nPP:
        PPxloc = extent[1] * xratio[1:-1] + offset*2
        PPyloc = extent[3] * yratio[1:-1] + offset
        
        pp_xy_proposal = [(x, y) for x in PPxloc for y in PPyloc]
        
        pp_cid_proposal = np.zeros(len(pp_xy_proposal))
        for i, point in  enumerate(pp_xy_proposal):
            pp_cid_proposal[i] = ixs.intersect(Point(point)).cellids.astype(int)[0]
        
        common_cells = np.intersect1d(pp_cid_proposal, blocked_cid)
        
        pp_cid_accepted = np.setdiff1d(pp_cid_proposal, common_cells)
        
        if len(pp_cid_accepted) < nPP:
            offset = np.random.randn() * dx_x

    
    return pp_cid_accepted.astype(int), pp_xy_accepted.astype(int)








