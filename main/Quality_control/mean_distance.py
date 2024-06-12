import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sys
sys.path.append('..')
from dependencies.model_params import get
from dependencies.load_template_model import load_template_model
from dependencies.create_pilot_points import create_pilot_points, create_pilot_points_even
from dependencies.plotting.plot_k_fields import plot_k_fields
from dependencies.create_k_fields import create_k_fields


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
num_neighbors = 4  # Number of closest neighbors

# CONSPIRACY??
pp_cid1, pp_xy1, near_dist1 = create_pilot_points(gwf, pars)
test_field1 = create_k_fields(gwf, pars, pp_xy = pp_xy1, pp_cid = pp_cid1, random = False, conditional = False)
mean_dist1 = mean_distance_to_closest_neighbors(pp_xy1, num_neighbors)

pp_cid2, pp_xy2, near_dist2 = create_pilot_points_even(gwf, pars)
test_field2 = create_k_fields(gwf, pars, pp_xy = pp_xy2, pp_cid = pp_cid2, random = False, conditional = False)
mean_dist2 = mean_distance_to_closest_neighbors(pp_xy1, num_neighbors)
# Example usage:


plot_k_fields(gwf, pars, [test_field1[0], test_field1[0]], points = pp_xy1)
plot_k_fields(gwf, pars, [test_field2[0], test_field2[0]], points = pp_xy2)

# plot_k_fields(gwf, pars, [gwf.npf.k.array, gwf.npf.k.array], points = pp_xy)


print("Mean distance to closest", num_neighbors, "neighbors for", len(pp_xy1), "random points in a", mg.extent[1], "x", mg.extent[3], "rectangle:", mean_dist1)
