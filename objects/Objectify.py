import numpy as np
import flopy 
from joblib import Parallel, delayed
from gstools import krige
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *

class Ensemble:
    def __init__(self, X, Ysim, obsloc, PPcor, htable, hvartable, tstp, meanh, varh, meank, Ens_PP, ncores):
        self.ncores = ncores
        self.X      = X
        self.nreal  = X.shape[1]
        self.nX     = X.shape[0]
        self.Ysim   = Ysim
        self.nobs   = Ysim.shape[0] #CHECK whetehr this is true
        self.obsloc = obsloc
        self.PPcor  = PPcor
        self.htable = htable
        self.hvartab= hvartable
        self.tstp   = tstp
        self.meanh  = meanh
        self.varh   = varh
        self.meank  = meank
        self.Ens_PP = Ens_PP
        self.nPP    = Ens_PP.shape[0] #CHECK whetehr this is true
        self.members = []
        
    def update_tstp(self):
        self.tstp += 1 
        
    # Add Member to Ensemble
    def add_member(self, member):
        self.members.append(member)
        
    def remove_member(self, member, j):
        self.members.remove(member)
        np.delete(self.X,       j, axis = 1)
        np.delete(self.Ysim,    j, axis = 2)
        np.delete(self.Ens_PP,  j, axis = 1)
        np.delete(self.Yobs,    j, axis = 1)
        self.nreal -= 1
    
    def initial_conditions(self, EvT, Rch, Qpu):
        # Only works with the preset member class
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                EvT, Rch, Qpu) for member in self.members
                )
        
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                self.members[j].set_hfield(np.squeeze(result[j]))
                        
                j = j + 1
        
    # Propagate Entire Ensemble
    def predict(self, EvT, Rch, Qpu):
        # Only works with the preset member class
        result = Parallel(
            n_jobs=self.ncores)(delayed(member.predict)(
                EvT, Rch, Qpu) for member in self.members
                )
        
        j = 0
        while j < self.nreal:
            if np.any(result[j]) == None:
                self.remove_member(self.members[j], j)
                print("Another ensemble member untimely laid down its work")
            else:
                # htable refers to the mean head level in the ensemble
                #self.htable[j,:,:]   = np.squeeze(result[j])
                self.X[self.nPP:,j]  = np.ndarray.flatten(result[j])
                
                for k in range(self.nobs):
                    self.Ysim[k,j]  = self.members[j].hfield[self.obsloc[k][0],self.obsloc[k][1]]
                        # self.Ysim[k,j]  = self.htable[j,self.obsloc[k][0],self.obsloc[k][1]]
                        
                j +=  1
    
    def update_hmean(self):
        newmean = np.zeros(self.meanh.shape)
        for member in self.members:
            newmean += member.hfield
        self.meanh = newmean / self.nreal
        self.htable[self.tstp-1,:,:] = self.meanh
        
    def update_kmean(self):
        newmean = np.zeros(self.meank.shape)
        for member in self.members:
            newmean += member.kfield
        self.meank = newmean / self.nreal
        
    def update_hvar(self):
        self.varh = np.reshape(np.var(self.X[self.nPP:],axis = 1),self.meanh.shape)
        self.hvartab[self.tstp-1,:,:] = self.varh
    
    
    def update_PP(self, PPk):
        for j in range(self.nreal):
            self.X[0:self.nPP, j] = PPk[:,j]
               
    def analysis(self, eps):
        
        # Compute mean of postX and Y_sim
        Xmean   = np.tile(np.array(np.mean(self.X, axis = 1)).T, (self.nreal, 1)).T
        Ymean   = np.tile(np.array(np.mean(self.Ysim,  axis  = 1)).T, (self.nreal, 1)).T
        
        # Fluctuations around mean
        X_prime = self.X - Xmean
        Y_prime = self.Ysim  - Ymean
        
        # Variance inflation
        # priorX  = X_prime * 1.01 + Xmean
        
        # Measurement uncertainty matrix
        R       = np.identity(self.nobs) * eps 
        
        # Covariance matrix
        Cyy     = 1/(self.nreal-1)*np.matmul((Y_prime),(Y_prime).T) + R 
                        
        return X_prime, Y_prime, Cyy
    
    def Kalman_update(self, damp, X_prime, Y_prime, Cyy, Y_obs):
        
        self.X += 1/(self.nreal-1) * (damp *
                    np.matmul(
                        X_prime, np.matmul(
                            Y_prime.T, np.matmul(
                                np.linalg.inv(Cyy), (Y_obs - self.Ysim)
                                )
                            )
                        ).T
                    ).T
        
        for j in range(len(self.members)):
            self.members[j].set_hfield(np.reshape(self.X[self.nPP:,j],self.members[j].hfield.shape))
        
        self.update_hmean()
        self.update_hvar()
        
    def PP_Kriging(self, cov_mod, PP_K, X, Y):
        
        Parallel(n_jobs=self.ncores)(delayed(self.members[j].updateK)(
                cov_mod, self.PPcor, PP_K[:,j], X, Y) for j in range(len(self.members))
                )
        
        self.update_kmean()
        
    
    def plot(self, mask, X, Y, true_heads, obscor, K_true):
        # ================ BEGIN FIRST FIGURE =====================================
        obsy, obsx = zip(*obscor)
        PPx, PPy = zip(*self.PPcor)
        fig1, axes1 = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        ax11, ax12, ax13 = axes1
        
        #First Subplot
        ax11.set_title("Mean, difference, and variance head in period " + str(int(self.tstp)))
        max_h = np.max(np.ma.masked_where(mask, self.meanh))
        cp11 = ax11.pcolor(
            X,Y,np.ma.masked_where(mask, self.meanh),
            cmap='RdBu', 
            vmin=np.min(np.ma.masked_where(mask, self.meanh)),  
            vmax = max_h
            )
        contours = ax11.contour(
            X, Y, self.meanh, levels=np.arange(
                np.min(self.meanh), max_h, (max_h-np.min(self.meanh))/50
                ), colors="black"
            )
        ax11.clabel(contours, fmt="%2.1f")
        ax11.set_aspect('equal', 'box')
        cax11 = plt.axes([0.83, 0.68, 0.035, 0.2])
        plt.colorbar(cp11, cax=cax11)
        

        # Second Subplot
        Diffh = np.ma.masked_where(mask, (true_heads - self.meanh))
        maxh_diff = np.max(np.abs(Diffh))
        cp12 = ax12.pcolor(
            X,Y,np.ma.masked_where(mask, Diffh), 
            cmap='RdBu', vmin=np.min(-maxh_diff), vmax=np.max(maxh_diff)
            )
        ax12.set_aspect('equal', 'box')
        cax12 = plt.axes([0.83, 0.4, 0.035, 0.2])
        plt.colorbar(cp12, cax=cax12)
        
        
        # Third Subplot
        maxh_var = np.max(np.abs(np.ma.masked_where(mask,self.varh)))
        cp13 = ax13.pcolor(
            X, Y, np.ma.masked_where(mask, self.varh),
            cmap='Reds', vmin=0, vmax=np.max(maxh_var)
            )
        ax13.set_aspect('equal', 'box')
        ax13.scatter(obsx, obsy, 15,'k','x')
        cax13 = plt.axes([0.83, 0.12, 0.035, 0.2])
        plt.colorbar(cp13, cax=cax13)
        
        plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)
        
        # ================ END FIRST FIGURE =======================================
        
        # ================ BEGIN SECOND FIGURE ====================================
        fig2, axes2 = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        ax21, ax22, ax23 = axes2
        
        # max_lnK_true = np.max(np.ma.masked_where(mask, K_true))
        # min_lnK_true = np.min(np.ma.masked_where(mask, K_true))
        # max_lnK_ens  = np.max(np.ma.masked_where(mask, self.meank))
        # min_lnK_ens  = np.min(np.ma.masked_where(mask, self.meank))
        
        # minK = np.min((min_lnK_true, min_lnK_ens))
        # maxK = np.max((max_lnK_true, max_lnK_ens))
        
        # First Subplot
        ax21.set_title("Mean, true, and variance log K in period " + str(int(self.tstp)))
        cp21 = ax21.pcolor(
            X,Y,np.ma.masked_where(mask, self.meank/2.3),
            cmap='RdBu', 
            vmin=1,  
            vmax=3.3
            )
        ax21.set_aspect('equal', 'box')
        ax21.scatter(PPx, PPy, 15,'k','x')
        cax21 = plt.axes([0.81, 0.68, 0.035, 0.2])
        plt.colorbar(cp21, cax=cax21)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        
        # Second Subplot
        cp22 = ax22.pcolor(
            X, Y, np.ma.masked_where(mask, K_true)/2.3,
            cmap='RdBu', 
            vmin=1, 
            vmax=3.3
            )
        ax22.set_aspect('equal', 'box')
        ax22.scatter(PPx, PPy, 15,'k','x')
        cax22 = plt.axes([0.81, 0.4, 0.035, 0.2])
        plt.colorbar(cp22, cax=cax22)
        
        # Third Subplot
        lnKrel = self.meank / K_true
        cp23 = ax23.pcolor(
            X, Y, np.ma.masked_where(mask, lnKrel),
            cmap='Reds', vmin=0, vmax=2
            )
        ax23.set_aspect('equal', 'box')
        ax23.scatter(PPx,PPy, 15,'k','x')
        cax23 = plt.axes([0.81, 0.12, 0.035, 0.2])
        plt.colorbar(cp23, cax=cax23)
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        # ================ END SECOND FIGURE ======================================     
        
        plt.show()
        
    def movie(self, mask, X, Y, true_heads, obscor, K_true):
    
        # ================ BEGIN FIRST FIGURE =====================================
        obsy, obsx = zip(*obscor)
        PPx, PPy = zip(*self.PPcor)
        fig1, axes1 = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        ax11, ax12, ax13 = axes1
        fig2, axes2 = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
        ax21, ax22, ax23 = axes2
    
    
        # ================ END SECOND FIGURE ======================================     
    
        def animate(i):
            ax11.clear()
            ax12.clear()
            ax13.clear()
            ax21.clear()
            ax22.clear()
            ax23.clear()
       
            #First Subplot
            ax11.set_title("Mittlerer GW-Spiegel, Differenz zur Wahrheit und Varianz in Zeitschritt " + str(i+1))
            max_h = np.max(np.ma.masked_where(mask, self.htable[i,:,:]))
            cp11 = ax11.pcolor(
                X,Y,np.ma.masked_where(mask, self.htable[i,:,:]),
                cmap='RdBu', 
                vmin = 9.5,  
                vmax = 14
                # vmin=np.min(np.ma.masked_where(mask, self.htable[i,:,:])),  
                # vmax = max_h
                )
            contours = ax11.contour(
                X, Y, self.htable[i,:,:], levels=np.arange(
                    np.min(self.htable[i,:,:]), max_h, (max_h-np.min(self.htable[i,:,:]))/50
                    ), colors="black"
                )
            ax11.clabel(contours, fmt="%2.1f")
            ax11.set_aspect('equal', 'box')
            cax11 = plt.axes([0.83, 0.68, 0.035, 0.2])
            plt.colorbar(cp11, cax=cax11)
       

            # Second Subplot
            Diffh = np.ma.masked_where(mask, (true_heads[i,:,:] - self.htable[i,:,:]))
            maxh_diff = np.max(np.abs(Diffh))
            cp12 = ax12.pcolor(
                X,Y,np.ma.masked_where(mask, Diffh), 
                cmap='RdBu', 
                vmin = -0.1, 
                vmax = 0.1
                # vmin=np.min(-maxh_diff), 
                # vmax=np.max(maxh_diff)
                )
            ax12.set_aspect('equal', 'box')
            cax12 = plt.axes([0.83, 0.4, 0.035, 0.2])
            plt.colorbar(cp12, cax=cax12)
       
       
            # Third Subplot
            maxh_var = np.max(np.abs(np.ma.masked_where(mask,self.hvartab[i,:,:])))
            cp13 = ax13.pcolor(
                X, Y, np.ma.masked_where(mask, self.hvartab[i,:,:]),
                cmap='Reds', 
                vmin=0, 
                vmax= 0.01
                # vmax=np.max(maxh_var)
                )
            ax13.set_aspect('equal', 'box')
            ax13.scatter(obsx, obsy, 15,'k','x')
            cax13 = plt.axes([0.83, 0.12, 0.035, 0.2])
            plt.colorbar(cp13, cax=cax13)
       
            plt.subplots_adjust(bottom=0.1, right=0.85, top=0.9)
       
            # ================ END FIRST FIGURE =======================================
       
            # ================ BEGIN SECOND FIGURE ====================================
              
            # max_lnK_true = np.max(np.ma.masked_where(mask, K_true))
            # min_lnK_true = np.min(np.ma.masked_where(mask, K_true))
            # max_lnK_ens  = np.max(np.ma.masked_where(mask, self.meank))
            # min_lnK_ens  = np.min(np.ma.masked_where(mask, self.meank))
       
            # minK = np.min((min_lnK_true, min_lnK_ens))
            # maxK = np.max((max_lnK_true, max_lnK_ens))
       
            # First Subplot
            ax21.set_title("Mittlere, wahre und Varianz von log(k) in Zeitschritt " + str(int(self.tstp)))
            cp21 = ax21.pcolor(
                X,Y,np.ma.masked_where(mask, self.meank/2.3),
                cmap='RdBu', 
                vmin = 1,  
                vmax = 3.3
                )
            ax21.set_aspect('equal', 'box')
            ax21.scatter(PPx, PPy, 15,'k','x')
            cax21 = plt.axes([0.81, 0.68, 0.035, 0.2])
            plt.colorbar(cp21, cax=cax21)
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
       
            # Second Subplot
            cp22 = ax22.pcolor(
                X, Y, np.ma.masked_where(mask, K_true)/2.3,
                cmap='RdBu', 
                vmin= 1, 
                vmax= 3.3
                )
            ax22.set_aspect('equal', 'box')
            ax22.scatter(PPx, PPy, 15,'k','x')
            cax22 = plt.axes([0.81, 0.4, 0.035, 0.2])
            plt.colorbar(cp22, cax=cax22)
       
            # Third Subplot
            lnKrel = self.meank / K_true
            cp23 = ax23.pcolor(
                X, Y, np.ma.masked_where(mask, lnKrel),
                cmap='Reds', vmin=0, vmax=2
                )
            ax23.set_aspect('equal', 'box')
            ax23.scatter(PPx,PPy, 15,'k','x')
            cax23 = plt.axes([0.81, 0.12, 0.035, 0.2])
            plt.colorbar(cp23, cax=cax23)
            plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    
        ani1 = animation.FuncAnimation(fig1, animate, 350, interval=50, blit=False)
        ani2 = animation.FuncAnimation(fig1, animate, 350, interval=50, blit=False)

        writer = animation.writers['ffmpeg'](fps=7)
    
        dpi = 250
        ani1.save('demo1.mp4',writer=writer,dpi=dpi)
        ani2.save('demo2.mp4',writer=writer,dpi=dpi)
       
    
class Member:
        
    def __init__(self, direc,  kfield, hfield, mname):
        self.direc      = direc
        self.hdirec     = direc + "/flow_output/flow.hds"
        self.kfield     = kfield
        self.hfield     = hfield
        self.mname      = mname
        self.sim        = flopy.mf6.modflow.MFSimulation.load(
                                mname, 
                                version             = 'mf6', 
                                exe_name            = 'mf6',
                                sim_ws              = direc, 
                                verbosity_level     = 0
                                )
    
    def get_hfield(self):
        return self.hfield
    
    def get_kfield(self):
        return self.kfield
        
    def set_kfield(self, Kf):
        assert Kf.shape == self.kfield.shape, "Why you change size of field?"
        
        # Kf[self.wellloc[0],self.wellloc[1]] = 11.366 #we need great equalizer
        self.kfield = Kf
        #self.set_PPk(Kf[PPloc])
        
        # Update Package
        mdl     = self.sim.get_model(self.mname)
        npf     = mdl.get_package("npf")
        
        npf.k.set_data(np.exp(Kf))
            
    def set_hfield(self, Hf):
        assert Hf.shape == self.hfield.shape, "Why you change size of field?"
        
        self.hfield = Hf
        
        # Update Package
        mdl     = self.sim.get_model(self.mname)
        ic      = mdl.get_package("ic")
         
        ic.data_list[0].set_data(Hf)
        
    def predict(self, EvT, Rch, Qpu):
        
        mdl     = self.sim.get_model(self.mname)
        
        evt     = mdl.get_package("evta")
        rch     = mdl.get_package("rcha")
        wel     = mdl.get_package("wel")
        
        evt.rate.set_data(EvT)
        rch.recharge.set_data(Rch)
        wel_data = wel.stress_period_data.get_data()
        wel_data[0]['q'][0] = Qpu[0]
        wel_data[0]['q'][1] = Qpu[1]
        wel.stress_period_data.set_data(wel_data)
    
        success, buff = self.sim.run_simulation()
        
        if not success:
            print(f"Model in {self.direc} has failed")
            Hf = None
               
        else:
            Hf = flopy.utils.binaryfile.HeadFile(self.hdirec).get_data(kstpkper=(0, 0))
            # self.set_hfield(self, Hf)
            
        return Hf
    
    def updateK(self, cov_mod, PPcor, PP_K, X, Y):
        
        krig = krige.Ordinary(cov_mod, cond_pos=PPcor, cond_val = PP_K)
        field = krig([X,Y])
        self.set_kfield(np.reshape(field[0],self.kfield.shape))

    
    
class VirtualReality(Member):
    # Inherit Properties of the Member class
    def __init__(self, direc,  obsloc, kfield, hfield, mname):
        super().__init__(direc, kfield, hfield, mname)
        self.obsloc = obsloc
        
    def set_kfield(self, Kf):
        assert Kf.shape == self.kfield.shape, "Why you change size of field?"
        
        #assert Kf[self.wellloc] == 11.366, "The great equalizer is not in place"
        
        # Check how many layers are around? --> flexible
        k_new = np.array([np.exp(Kf),np.exp(Kf)/5])
        k_new[0,54,71]    = 86400
        k_new[0,28,190]   = 86400
        k_new[1,54,71]    = 86400
        k_new[1,28,190]   = 86400
        self.kfield = k_new
        #self.set_PPk(Kf[PPloc])
        
        # Update Package
        mdl     = self.sim.get_model(self.mname)
        npf     = mdl.get_package("npf")
        
        npf.k.set_data(k_new)
        
    def predict(self, EvT, Rch, Qpu):
        
        mdl     = self.sim.get_model(self.mname)
        
        evt     = mdl.get_package("evta")
        rch     = mdl.get_package("rcha")
        wel     = mdl.get_package("wel")
        
        evt.rate.set_data(EvT)
        rch.recharge.set_data(Rch)
        wel_data = wel.stress_period_data.get_data()
        # Are we pumping twice the amount in VR??
        wel_data[0]['q'][0] = Qpu[0]/2
        wel_data[0]['q'][1] = Qpu[1]/2
        wel.stress_period_data.set_data(wel_data)
    
        success, buff = self.sim.run_simulation()
        
        if not success:
            print("Virtual Truth has failed")
            Hf = None
               
        else:
            Hf = flopy.utils.binaryfile.HeadFile(self.hdirec).get_data(kstpkper=(0, 0))
            # self.set_hfield(self, Hf)
            
        self.set_hfield(Hf)
        
    def pert_obs(self, nreal, nobs, eps):
        # Entering observation data for all ensemble members (allows individual pert)
        obs = [np.mean(self.hfield[:,self.obsloc[i][0], self.obsloc[i][1]]) for i in range(len(self.obsloc))]
        
        Y_obs_pert = np.tile(obs, (nreal,1)).T + np.random.normal(loc=0, scale=eps, size=(nobs, nreal))
        
        return Y_obs_pert