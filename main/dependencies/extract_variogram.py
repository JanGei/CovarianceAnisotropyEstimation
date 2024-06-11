import gstools as gs
import numpy as np
import matplotlib.pyplot as plt

def extract_vario(x,y,field):
    
    bins = np.arange(40)
    bin_center, gamma = gs.vario_estimate((x, y), field, bins)
    models = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Matern": gs.Matern,
    }
    scores = {}
    
    # plot the estimated variogram
    plt.scatter(bin_center, gamma, color="k", label="data")
    ax = plt.gca()
    
    # fit all models to the estimated variogram
    for model in models:
        fit_model = models[model](dim=2)
        para, pcov, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True)
        fit_model.plot(x_max=40, ax=ax)
        scores[model] = r2
    
    ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    print("RANKING by Pseudo-r2 score")
    for i, (model, score) in enumerate(ranking, 1):
        print(f"{i:>6}. {model:>15}: {score:.5}")
    
    plt.show()
    
    
    
    
    
    
    
    
    # angle = np.pi / 8
    # bins = range(0, 40, 2)
    # bin_center, dir_vario, counts = gs.vario_estimate(
    #     *((x, y), field, bins),
    #     direction=gs.rotated_main_axes(dim=2, angles=angle),
    #     angles_tol=np.pi / 16,
    #     bandwidth=8,
    #     mesh_type="structured",
    #     return_counts=True,
    # )
    
    # print("Original:")
    # print(f'{orig[0]}(dim=2, var={orig[1]}, len_scale=={orig[2]}, nugget=0.0, anis={orig[3]}, angles={orig[4]}')
    # model.fit_variogram(bin_center, dir_vario)
    # print("Fitted:")
    # print(model)