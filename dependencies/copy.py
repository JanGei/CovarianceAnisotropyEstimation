import os
import shutil

def create_Ensemble(orig_dir: str, model_dir: str, n_members: int) -> list:
    ens_m_dir = []
    for i in range(n_members):
        mem_dir = model_dir + f'{i}'
        # Check if the destination folder already exists
        if os.path.exists(mem_dir):
            # Remove the existing destination folder and its contents
            shutil.rmtree(mem_dir)
    
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
