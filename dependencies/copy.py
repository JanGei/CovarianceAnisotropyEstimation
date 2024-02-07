import os
import shutil

def create_Ensemble(pars: dict) -> list:
    ens_m_dir = []
    orig_dir    = pars['tm_ws']
    mem_ws      = pars['mem_ws']
    n_mem       = pars['n_mem']
    ens_ws      = pars['ens_ws']
    
    # removing old members
    directories = [d for d in os.listdir(ens_ws) if os.path.isdir(os.path.join(ens_ws, d))]
    for d in directories:
        if d.startswith('member'):
            shutil.rmtree(os.path.join(ens_ws, d))
        
    for i in range(n_mem):
        mem_dir = mem_ws + f'{i}'
    
        # Copy the model folder to new folder
        shutil.copytree(orig_dir, mem_dir)
        ens_m_dir.append(mem_dir)
        
    return ens_m_dir
    
def copy_model(orig_dir:str, model_dir: str) -> None:
    
    # Check if the destination folder already exists
    if os.path.exists(model_dir):
        # Remove the existing destination folder and its contents
        shutil.rmtree(model_dir)

    # Copy the model folder to new folder
    shutil.copytree(orig_dir, model_dir)
