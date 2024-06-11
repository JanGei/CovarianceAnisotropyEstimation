import flopy
import numpy as np

class Virtual_Reality:
    
    def __init__(self,  pars):
        self.direc      = pars['trs_ws']
        self.mname      = pars['mname']
        self.pars       = pars
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = pars['trs_ws'], 
                                verbosity_level     = 0
                                )
        self.gwf        = self.sim.get_model(self.mname)
        self.npf        = self.gwf.npf
        self.rch        = self.gwf.rch
        self.riv        = self.gwf.riv
        self.wel        = self.gwf.wel
        self.ic         = self.gwf.ic
        self.chd        = self.gwf.chd
        self.mg         = self.gwf.modelgrid
        self.cxy        = np.vstack((self.mg.xyzcellcenters[0], self.mg.xyzcellcenters[1])).T
        self.dx         = pars['dx']

            
        
    def set_field(self, field, pkg_name: list):
        for i, name in enumerate(pkg_name):
            if name == 'npf':
                self.old_npf =  self.npf.k.get_data()
                self.npf.k.set_data(np.reshape(field[i],self.npf.k.array.shape))
                self.npf.write()
            elif name == 'rch':
                self.rch.stress_period_data.set_data(field[i])
                self.rch.write()
            elif name == 'riv':
                self.riv.stress_period_data.set_data(field[i])
                self.riv.write()
            elif name == 'wel':
                self.wel.stress_period_data.set_data(field[i])
                self.wel.write()
            elif name == 'h':
                self.ic.strt.set_data(field[i])
                self.ic.write()
            else:
                print(f'The package {name} that you requested is not part of the model')
            
        
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
            else:
                print(f'The package {name} that you requested is not part of the model')
                
        return fields
    
    def update_transient_data(self,rch_data, wel_data, riv_data):
        
        spds = self.get_field(['rch', 'wel', 'riv'])
        
        rch_spd = spds['rch']
        wel_spd = spds['wel']
        riv_spd = spds['riv']

        rivhl = np.ones(np.shape(riv_spd[0]['cellid']))
        
        rch_spd[0]['recharge'] = rch_data
        riv_spd[0]['stage'] = rivhl * riv_data
        wel_spd[0]['q'] = wel_data
        
        self.set_field([rch_spd, wel_spd, riv_spd],['rch', 'wel', 'riv']) 
        
    def simulation(self):
        success, buff = self.sim.run_simulation()
        if not success:
            import sys
            print('The Virtual Reality did crash - Aborting')
            sys.exit()
        
       
    def update_ic(self):
        h_field = self.get_field('h')['h']
        self.ic.strt.set_data(h_field)
        self.ic.write()
        return h_field
    
    def get_observations(self, obs_cid):
        h = self.get_field(['h'])['h'].flatten()
        #perturb these measurements individually for every ensemble member - EnKF?
        ny = len(obs_cid)
        Ymeas = np.zeros((ny,1))
        for i in range(ny):
            Ymeas[i,0] = h[obs_cid[i]]
        
        return Ymeas