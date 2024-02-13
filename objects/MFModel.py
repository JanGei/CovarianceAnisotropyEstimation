import flopy
import gstools as gs
from gstools import krige
import numpy as np

class MFModel:
    
    def __init__(self, direc,  mname, cov_data, cov_model):
        self.direc      = direc
        self.mname      = mname
        self.cov_data   = cov_data
        self.cov_model  = cov_model
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = direc, 
                                verbosity_level     = 0
                                )
        self.gwf        = self.sim.get_model(mname)
        self.npf        = self.gwf.npf
        self.rch        = self.gwf.rch
        self.riv        = self.gwf.riv
        self.wel        = self.gwf.wel
        self.ic         = self.gwf.ic
        self.chd        = self.gwf.chd
        self.cell_xy    = self.gwf.modelgrid.xyzcellcenters
        
        
    def set_field(self, field, pkg_name: list):
        for i, name in enumerate(pkg_name):
            if name == 'npf':
                self.npf.k.set_data(np.reshape(field[i],self.npf.k.array.shape))
            elif name == 'rch':
                # rch_spd = self.rch.stress_period_data.get_data()
                # rch_spd[0]['recharge'] = field[i]
                self.rch.stress_period_data.set_data(field[i])
            elif name == 'riv':
                # riv_spd = self.riv.stress_period_data.get_data()
                # riv_spd[0]['stage'] = field[i]
                self.riv.stress_period_data.set_data(field[i])
            elif name == 'wel':
                # wel_spd = self.wel.stress_period_data.get_data()
                # wel_spd[0]['q'] = field[i]
                self.wel.stress_period_data.set_data(field[i])
            elif name == 'h':
                self.ic.strt.set_data(field[i])
            else:
                print(f'The package {name} that you requested is not part ofthe model')
            
        self.sim.write_simulation()
        
    def get_field(self, pkg_name: list) -> dict:
        fields = {}
        for name in pkg_name:
            if name == 'npf':
                fields.update({name:self.npf.k.get_data()})
            elif name == 'rch':
                fields.update({name:self.rch.stress_period_data.get_data()})
            elif name == 'riv':
                fields.update({name:self.riv.stress_period_data.get_data()})
            elif name == 'wel':
                fields.update({name:self.wel.stress_period_data.get_data()})
            elif name == 'chd':
                fields.update({name:self.chd.stress_period_data.get_data()})
            elif name == 'h':
                fields.update({name:self.gwf.output.head().get_data()})
            elif name == 'cov_data':
                fields.update({name:self.cov_data})
            else:
                print(f'The package {name} that you requested is not part ofthe model')
                
        return fields
                
    def simulation(self):
        self.sim.run_simulation()
        
    def update_ic(self):
        self.ic.strt.set_data(self.get_field('h')['h'])
        self.sim.write_simulation()
        
        
    def kriging(self, params, data, pp_xy):
        
        if 'cov_data' in params:
            self.cov_model.len_scale = [data[0][0], data[0][1]]
            self.cov_model.angles = data[0][2]
            self.cov_model.var = data[0][3]

            pp_k = data[1]
                
        else:
            pp_k = data[0]
                
        
        krig = krige.Ordinary(self.cov_model, cond_pos=(pp_xy[:,0], pp_xy[:,1]), cond_val = np.log(pp_k))
        field = krig((self.cell_xy[0], self.cell_xy[1]))
        
        self.set_field([np.exp(field[0])], ['npf'])
        
        
        