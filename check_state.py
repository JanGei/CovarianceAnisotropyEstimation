import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

def prepare_data_end(directory, pars, krig = False, ellips = True, new = True, end = False):   
    
    out_dir = os.path.join(directory, 'output')
    
    if ellips:
        files = os.listdir(out_dir)
        matching_files = [file for file in files if file.startswith('covariance_model')]
    
        data = []
        for i in range(len(matching_files)):
            data.append(np.loadtxt(os.path.join(out_dir, matching_files[i])))
    
    if data == []:
        ellips = False
        ellipsis_data = []
        mean_data = []
    else:
        ellipsis_data = np.stack(data, axis = 1)
        mean_data = np.mean(ellipsis_data, axis = 1)
    
    errors = np.loadtxt(os.path.join(out_dir, 'errors.dat')) 

    return ellipsis_data, mean_data, errors, ellips

if __name__ == '__main__':
    
    # pars = get()
    cwd = os.getcwd()
    # specify which folder to investigate
    folder = input('Folder Name: ')
    folders = [folder for folder in os.listdir(os.path.join(cwd, folder))]
    # Substrings to remove
    substrings = [".sh", ".py", "dependencies"]
    

    models = [folder for folder in os.listdir(os.path.join(cwd, folder))]
    print(models)
    models = [element for element in models if 'main' not in element]
    print(models)
    for model in models:
        target_folder = os.path.join(cwd, folder, model)
        # model_directory = os.path.join(target_folder, 'output')
        
        krig = True
        # Test does not work atm
        test = True
        
        dep_dir = os.path.join(cwd, target_folder, 'dependencies')
        module_name1 = f"{dep_dir}.model_params"
        spec1 = importlib.util.spec_from_file_location(module_name1, os.path.join(dep_dir, "model_params.py"))
        module1 = importlib.util.module_from_spec(spec1)
        spec1.loader.exec_module(module1)
        pars = module1.get()
            
        ellipsis_data, mean_data, errors, ellips = prepare_data_end(target_folder, pars, new = False, end = True)
        
        print(ellipsis_data.shape())
        l = np.max(pars['lx'][0] * 1.5)
        x = np.linspace(-l, l, 300)
        y = np.linspace(-l, l, 300)
        X, Y = np.meshgrid(x, y)

        fig, ax = plt.subplots(figsize=(9, 9))
        
        for cov in ellipsis_data[-1,:,:]:
            M = np.array(([cov[0], cov[1]], [cov[1], cov[2]]))
            res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
            ax.contour(X, Y, res, levels=[0], colors='black', alpha=0.2)

        M = np.array(([mean_data[-1,:][0], mean_data[-1,:][1]], [mean_data[-1,:][1], mean_data[-1,:][2]]))
        res = X**2 * M[0,0] + X*Y*(M[0,1] + M[1,0]) + Y**2 * M[1,1] - 1
        ax.contour(X, Y, res, levels=[0], colors='blue', alpha=0.5)

        ellipse = patches.Ellipse((0, 0), pars['lx'][0][0]*2, pars['lx'][0][1]*2, angle=pars['ang'][0], fill=False, color='red')
        ax.add_patch(ellipse)

        ax.set_aspect('equal', 'box')
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)
        plt.grid(True)

        # Remove axis labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title(f'Time Step ({len(ellipsis_data)}) for model {model}')
        ax.text(l/4*3, l/4*3, f'OLE  {errors[-1,:][0]} \nTE-1 {errors[-1,:][1]}\nTE-2 {errors[-1,:][2]}', fontsize=9, color='black')

        plt.show()
        