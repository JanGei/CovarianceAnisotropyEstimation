import flopy
# import gstools as gs
from gstools import krige
import numpy as np

class MFModel:
    
    def __init__(self, direc: str,  mname:str, cov_model, ellips):
        self.direc      = direc
        self.mname      = mname
        self.cov_model  = cov_model
        self.ellips_mat = np.array([[ellips[0], ellips[1]], [ellips[1], ellips[2]]])
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
                print(f'The package {name} that you requested is not part ofthe model')
            
        
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
                fields.update({name:np.array([self.ellips_mat[0,0],
                                              self.ellips_mat[0,1],
                                              self.ellips_mat[1,1]])})
            else:
                print(f'The package {name} that you requested is not part of the model')
                
        return fields
                
    def simulation(self):
        self.sim.run_simulation()
        
    def update_ic(self):
        self.ic.strt.set_data(self.get_field('h')['h'])
        self.ic.write()
        
        
    def kriging(self, params, data, pp_xy):
        
        if 'cov_data' in params:   
            
            # The normalized (unit “length”) eigenvectors, such that the column
            # eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
            eigenvalues, eigenvectors, mat, pos_def = self.check_new_matrix(data[0])
            
            if not pos_def:
                # Ansatz Olaf: Update so lange verkleinern, bis es passt und
                # die Matrix positiv definit ist
                difmat = mat - self.ellips_mat
                reduction = 0.96
                while reduction > 0:
                    test_mat = self.ellips_mat + reduction * difmat
                    eigenvalues, eigenvectors = np.linalg.eig(test_mat)
                    if np.all(eigenvalues > 0):
                        pos_def = True
                        break
                    else:
                        reduction -= 0.05
            
            if pos_def:
                self.update_ellips_mat(mat)
            
                l1 = 1 / np.sqrt(eigenvalues[0])
                l2 = 1 / np.sqrt(eigenvalues[1])
                # Get the rotation angle in radians
                angle = np.arccos(np.dot(np.array([1,0]), eigenvectors[:,0]))
                # Here, an eflection method is used to prevent negative corrl
                self.cov_model.len_scale = [l1, l2]
                # angle is must be in radians
                self.cov_model.angles = angle
                
                pp_k = data[1]
                
                krig = krige.Ordinary(self.cov_model,
                                      cond_pos = (pp_xy[:,0],
                                                  pp_xy[:,1]),
                                      cond_val = np.log(pp_k))
                field = krig((self.cell_xy[0],
                              self.cell_xy[1]))
                
                self.set_field([np.exp(field[0])], ['npf'])
                
            else:
                # If nothing works, keep old solution
                eigenvalues, eigenvectors = np.linalg.eig(self.ellips_mat)
                l1 = 1 / np.sqrt(eigenvalues[0])
                l2 = 1 / np.sqrt(eigenvalues[1])
                # Get the rotation angle in radians
                angle = np.arccos(np.dot(np.array([1,0]), eigenvectors[:,0]))
                
                
            return [l1, l2, angle]
                
        else:
            pp_k = data
            
            krig = krige.Ordinary(self.cov_model,
                                  cond_pos = (pp_xy[:,0],
                                              pp_xy[:,1]),
                                  cond_val = np.log(pp_k))
            field = krig((self.cell_xy[0],
                          self.cell_xy[1]))
            
            self.set_field([np.exp(field[0])], ['npf'])
        

    def update_ellips_mat(self, mat):
        self.ellips_mat = mat
        
    def check_new_matrix(self, data):
        mat = np.zeros((2,2))
        mat[0,0] = data[0]
        mat[0,1] = data[1]
        mat[1,0] = data[1]
        mat[1,1] = data[2]
        
        eigenvalues, eigenvectors = np.linalg.eig(mat)
        
        #check for positive definiteness
        if np.all(eigenvalues > 0):
            pos_def = True
        else:
            pos_def = False
            
        return eigenvalues, eigenvectors, mat, pos_def


        