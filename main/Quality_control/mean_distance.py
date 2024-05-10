import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sys
sys.path.append('..')
from dependencies.model_params import get
from dependencies.load_template_model import load_template_model
from dependencies.create_pilot_points import create_pilot_points
from dependencies.plotting.plot_k_fields import plot_k_fields


def mean_distance_to_closest_neighbors(points, num_neighbors=4):
    """
    Calculate the mean distance to the num_neighbors closest neighbors for n random points
    in a 2D plane with dimensions x and y.
    """
    distances = []

    # Calculate pairwise distances between all points
    pairwise_distances = cdist(points, points)

    # Exclude self-distances
    np.fill_diagonal(pairwise_distances, np.inf)

    # Find indices of num_neighbors closest neighbors for each point
    closest_indices = np.argsort(pairwise_distances, axis=1)[:, :num_neighbors]

    # Calculate mean distance to closest neighbors for each point
    for i in range(len(points)):
        closest_points = points[closest_indices[i]]
        point_distances = np.linalg.norm(closest_points - points[i], axis=1)
        mean_distance = np.mean(point_distances)
        distances.append(mean_distance)

    mean_distance = np.mean(distances)
    return mean_distance



pars = get()
sim, gwf = load_template_model(pars)
mg = gwf.modelgrid
pp_cid, pp_xy = create_pilot_points(gwf, pars)

# Example usage:
num_neighbors = 4  # Number of closest neighbors
mean_dist = mean_distance_to_closest_neighbors(pp_xy, num_neighbors)
plot_k_fields(gwf, pars, [gwf.npf.k.array, gwf.npf.k.array], points = pp_xy)


print("Mean distance to closest", num_neighbors, "neighbors for", len(pp_xy), "random points in a", mg.extent[1], "x", mg.extent[3], "rectangle:", mean_dist)
