import flopy
import numpy as np
# import os
import sys
sys.path.append('..')
from dependencies.randomK_points import randomK_points
from dependencies.covarmat_s import covarmat_s
# from dependencies.plotting.plot_k_fields import plot_k_fields
# import time


class MFModel:
    
    def __init__(self, direc: str,  pars, l_angs = [], ellips = []):
        self.direc      = direc
        self.mname      = pars['mname']
        self.pars       = pars
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = direc, 
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
        dxmax           = np.max([max(sublist) - min(sublist) for sublist in self.mg.xvertices])
        dymax           = np.max([max(sublist) - min(sublist) for sublist in self.mg.yvertices])
        self.dx         = [dxmax, dymax]
        self.old_npf    = []
        self.n_failure  = 0
        self.n_neg_def  = 0
        if pars['pilotp']:
            self.ellips_mat = np.array([[ellips[0], ellips[1]], [ellips[1], ellips[2]]])
            self.lx         = [l_angs[0], l_angs[1]]
            self.ang        = l_angs[2]
            
        
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
            elif name == 'cov_data':
                fields.update({name:np.array([self.ellips_mat[0,0],
                                              self.ellips_mat[0,1],
                                              self.ellips_mat[1,1]])})
            else:
                print(f'The package {name} that you requested is not part of the model')
                
        return fields
                
    def simulation(self):
        success, buff = self.sim.run_simulation()
        if not success:
            self.set_field([self.old_npf], ['npf'])
            self.sim.run_simulation()
            self.n_failure += 1

        
    def update_ic(self):
        self.ic.strt.set_data(self.get_field('h')['h'])
        self.ic.write()
        
        
    def kriging(self, params, data, pp_xy, pp_cid, mean_cov_par, var_cov_par):
        # start_time = time.time()
        if 'cov_data' in params:   
            
            # The normalized (unit “length”) eigenvectors, such that the column
            # eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
            eigenvalues, eigenvectors, mat, pos_def = self.check_new_matrix(data[0])
            
            if not pos_def:
                # Ansatz Olaf: Update so lange verkleinern, bis es passt und
                # die Matrix positiv definit ist
                reduction = 0.96
                difmat = mat - self.ellips_mat

                while reduction > 0:
                    test_mat = self.ellips_mat + reduction * difmat
                    eigenvalues, eigenvectors = np.linalg.eig(test_mat)
                    if np.all(eigenvalues > 0):
                        pos_def = True
                        break
                    else:
                        reduction -= 0.05
            
            if pos_def:
                success = True
                l1, l2, angle = self.pos_krig(mat, eigenvalues, eigenvectors, data, pp_cid, pp_xy)
                
            else:
                # If nothing works, keep old solution
                eigenvalues, eigenvectors = np.linalg.eig(self.ellips_mat)
                
                l1, l2, angle = self.extract_truth(eigenvalues, eigenvectors)
                success = False
                self.n_neg_def += 1                
                # If nothing has worked for 10 consecutive timesteps, draw a new
                # variogram from the mean variogram distribution
                if self.n_neg_def == 10:
                    pos_def = False
                    while not pos_def:
                        a = np.random.normal(mean_cov_par[0,0], np.sqrt(var_cov_par[0,0]))
                        m = np.random.normal(mean_cov_par[0,1], np.sqrt(var_cov_par[0,1]))
                        b = np.random.normal(mean_cov_par[1,1], np.sqrt(var_cov_par[1,1]))
                    
                        eigenvalues, eigenvectors, mat, pos_def = self.check_new_matrix([a,m,b])
                        
                    l1, l2, angle = self.pos_krig(mat, eigenvalues, eigenvectors, data, pp_cid, pp_xy)
                elif self.n_neg_def == 1:
                    print('A new model got stuck')
                    
            return [[l1, l2, angle%np.pi], self.ellips_mat, success]
                
        else:
            pp_k = data
          
            field = self.conditional_field(pp_xy, pp_cid, pp_k)
            
            self.set_field([np.exp(field)], ['npf'])
        
        # print(f'Entire function took {(time.time() - start_time):.2f} seconds')
    
    def extract_truth(self, eigenvalues, eigenvectors):
        
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
    
    def update_ellips_mat(self, mat):
        self.ellips_mat = mat.copy()
     
    def pos_krig(self, mat, eigenvalues, eigenvectors, data, pp_cid, pp_xy):
        self.update_ellips_mat(mat)
        
        l1, l2, angle = self.extract_truth(eigenvalues, eigenvectors)
        
        if l2 > l1:
            l1, l2 = l2, l1
            angle += np.pi/2
            
        self.lx = [l1, l2]
        self.ang = angle%np.pi
        
        # Is ppk really without a logarithm
        pp_k = data[1]
        
        field = self.conditional_field(pp_xy, pp_cid, pp_k)
        
        self.set_field([np.exp(field)], ['npf'])
        
        if self.n_neg_def > 0:
            if self.n_neg_def == 10:
                print('10 tries reached- replacing covariance model')
            else:
                print('A model has been fixed')
            self.n_neg_def = 0
        
        return l1, l2, angle
    
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

    def conditional_field(self, pp_xy, pp_cid, pp_k):
        
        sigma   = self.pars['sigma'][0]
        cov     = self.pars['cov']
        sig_meas = 0.1 # standard deviation of measurement error
        
        # random, unconditional field for the given variogram
        Kflat = np.log(randomK_points(self.mg.extent, self.cxy, self.dx,  self.lx, self.ang, sigma, cov, 1, self.pars)) 

        # Construct covariance matrix of measurement error
        m = len(pp_k)
        n = self.cxy.shape[0]
        # Discretized trend functions (for constant mean)
        X = np.ones((n,1))
        Xm = np.ones((m,1))        
        
        R = np.eye(m)* sig_meas**2
        
        Ctype = 2
        # Construct the necessary covariance matrices
        Qssm = covarmat_s(self.cxy,pp_xy,Ctype,[sigma,self.lx,self.ang]) 
        Qsmsm = covarmat_s(pp_xy,pp_xy,Ctype,[sigma,self.lx, self.ang])
        
        # kriging matrix and its inverse
        krigmat = np.vstack((np.hstack((Qsmsm+R, Xm)), np.append(Xm.T, 0)))
        # ikrigmat = np.linalg.inv(krigmat)
        
        # generating a conditional realisation
        sunc_at_meas = np.zeros(m)
        for ii in range(m):
            sunc_at_meas[ii] = Kflat[int(pp_cid[ii])] 
        
        # Perturb the measurements and subtract the unconditional realization
        spert = np.squeeze(pp_k) + np.squeeze(sig_meas * np.random.randn(*pp_k.shape)) - np.squeeze(sunc_at_meas)
        
        # Solve the kriging equation
        sol = np.linalg.lstsq(krigmat, np.append(spert.flatten(), 0), rcond=None)[0]
        
        # Separate the trend coefficient(s) from the weights of the covariance-functions in the function-estimate form
        xi = sol[:m]
        beta = sol[m]
        
        s_cond = np.squeeze(Qssm.dot(xi)) + np.squeeze(X.dot(beta)) + Kflat

        return s_cond
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        