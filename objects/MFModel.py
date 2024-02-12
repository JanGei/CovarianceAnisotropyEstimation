import flopy
import warnings
# Suppress DeprecationWarning temporarily
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
class MFModel:
    
    def __init__(self, direc,  mname, cov_data):
        self.direc      = direc
        self.mname      = mname
        self.cov_data   = cov_data
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
        
        
    def set_field(self, field, pkg_name: list):
        for i, name in enumerate(pkg_name):
            if name == 'npf':
                self.npf.k.set_data(field[i])
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
        
        
        
        
        
        