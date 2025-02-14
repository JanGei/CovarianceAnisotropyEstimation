import os
import shutil

def create_Ensemble(pars: dict, iso = False) -> list:
    n_mem       = pars['n_mem']
    # vr_dir      = pars['trs_ws']
    # ss_dir      = pars['sim_ws']
    ens_m_dir = []
    
    # removing old Ensemble
    if iso:
        e_ws = pars['sh_ens']
        m_ws = pars['sh_mem']
    else:
        e_ws = pars['ens_ws']
        m_ws = pars['mem_ws']
        
    if os.path.exists(e_ws) and os.path.isdir(e_ws):
        shutil.rmtree(e_ws)
        os.mkdir(e_ws)
    else:
        os.mkdir(e_ws)

    # create template model
    # shutil.copytree(vr_dir, pars['tm_ws'])
    
    for i in range(n_mem):
        mem_dir = m_ws + f'{i}'
        # Copy the steady_state model folder to new folders
        shutil.copytree(pars['sim_ws'], mem_dir)
        ens_m_dir.append(mem_dir)

        
    return ens_m_dir

def create_shadow_Ensemble(pars: dict):
    n_mem       = pars['n_mem']
    mem_ws      = pars['mem_ws']
    shadowmem_ws= pars['sh_mem']
    shadowens_ws= pars['sh_ens']
    shadow_ens_m_dir= []
    # Creating Ghost Ensemble
    if os.path.exists(shadowens_ws) and os.path.isdir(shadowens_ws):
        shutil.rmtree(shadowens_ws)
        os.mkdir(shadowens_ws)
    else:
        os.mkdir(shadowens_ws)
        
    for i in range(n_mem):
        mem_dir = mem_ws + f'{i}'
        shadowmem_dir = shadowmem_ws + f'{i}'
    
        # Copy the models to the ghost ensemble
        shutil.copytree(mem_dir, shadowmem_dir)
        shadow_ens_m_dir.append(shadowmem_dir)
        
    return shadow_ens_m_dir


def copy_model(orig_dir:str, model_dir: str) -> None:
    
    # Check if the destination folder already exists
    if os.path.exists(model_dir):
        # Remove the existing destination folder and its contents
        shutil.rmtree(model_dir)

    # Copy the model folder to new folder
    shutil.copytree(orig_dir, model_dir)

def create_Test_Mod(pars: dict) -> list:
    ens_m_dir = []
    mem_ws      = pars['mem_ws']
    ens_ws      = pars['ens_ws']
    vr_dir      = pars['trs_ws']
   
    # removing old Ensemble
    if os.path.exists(ens_ws) and os.path.isdir(ens_ws):
        shutil.rmtree(ens_ws)
        os.mkdir(ens_ws)
    else:
        os.mkdir(ens_ws)
    
    # create template model
    shutil.copytree(vr_dir, pars['tm_ws'])

    mem_dir = mem_ws + '_test'
    
    # Copy the model folder to new folder
    shutil.copytree(vr_dir, mem_dir)
    ens_m_dir.append(mem_dir)
        
    return ens_m_dir