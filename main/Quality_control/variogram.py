import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
# from dependencies.load_template_model import load_template_model
from dependencies.model_params import get
import os
from dependencies.load_template_model import load_template_model

def calculate_directional_variogram(df, lag_distances, direction, tolerance):
    """
    Calculate the directional variogram for a set of lag distances.

    :param df: DataFrame with columns 'x', 'y', 'value'
    :param lag_distances: List of lag distances to calculate variograms
    :param direction: Direction to look in degrees (0-360, where 0 is east)
    :param tolerance: Tolerance in degrees for direction
    :return: Dictionary with lag distances as keys and semi-variograms as values
    """
    coords = df[['x', 'y']].values
    values = df['value'].values
    variograms = {}

    # Compute distance matrix
    dist_matrix = distance_matrix(coords, coords)

    # Loop over each lag distance
    for lag in lag_distances:
        # Initialize variables
        semi_variances = []
        total_pairs = 0

        # Loop through each sample
        for i in range(len(df)):
            x_i, y_i = coords[i]
            value_i = values[i]
            
            # Find all samples within the specified distance range
            for j in range(len(df)):
                if i == j:
                    continue

                x_j, y_j = coords[j]
                value_j = values[j]

                # Calculate distance and angle
                dist = dist_matrix[i,j]
                if dist < lag:
                    # Calculate angle
                    angle = np.degrees(np.arctan2(y_j - y_i, x_j - x_i)) % 360
                    angle_diff = min(abs(angle - direction), abs(360 - abs(angle - direction)))

                    # Check if the angle is within tolerance
                    if angle_diff <= tolerance:
                        squared_diff = (value_i - value_j) ** 2
                        semi_variances.append(squared_diff)
                        total_pairs += 1

        # Compute semi-variogram for this lag distance
        if total_pairs > 0:
            semi_variogram = np.mean(semi_variances) / 2
            variograms[lag] = semi_variogram
        else:
            variograms[lag] = np.nan

    return variograms




pars        = get()
sim, gwf = load_template_model(pars)
c_xy = gwf.modelgrid.xyzcellcenters

k_ref = np.loadtxt(pars['k_r_d'])
pp_xy = np.loadtxt(os.path.join(pars['resdir'],'pp_xy.dat'))
pp_cid = np.loadtxt(os.path.join(pars['resdir'],'pp_cid.dat'))
k_pp = k_ref[pp_cid.astype(int)]


df = pd.DataFrame({'x': pp_xy[:,0],
                   'y': pp_xy[:,1],
                   'value': k_pp})

dflarge = pd.DataFrame({'x': c_xy[0],
                        'y': c_xy[1],
                        'value': k_ref})

# Extract coordinates and values
coords = df[['x', 'y']].values
values = df['value'].values

# Example usage
lag_distances = np.arange(200,1500, 100)
tolerance = 10  # Tolerance of 10 degrees
directions = np.arange(-90,91, 15)

for direction in directions:
    variograms = calculate_directional_variogram(dflarge, lag_distances, direction, tolerance)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(variograms.keys(), variograms.values(), marker='o', linestyle='-', color='b')
    plt.xlabel('Lag Distance (meters)')
    plt.ylabel('Semi-variogram')
    plt.ylim([0, 1e-6])
    plt.xlim([0,1000])
    plt.title(f'Directional Variogram for {direction}')
    plt.grid(True)
    plt.show()

