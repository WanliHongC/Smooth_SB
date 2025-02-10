import numpy as np 
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pandas as pd
import ot

# Compute Covariance Matrix 
def find_longest_consecutive_trajectory(traj):
    max_length = 1
    current_length = 1
    max_index = traj[0]
    current_index = traj[0]

    for i in range(1, len(traj)):
        if traj[i] == traj[i - 1]:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_index = current_index
            current_length = 1
            current_index = traj[i]

    # Check the last segment
    if current_length > max_length:
        max_length = current_length
        max_index = current_index

    return max_index, max_length

def find_closest_trajectory(traj, arr):
    T, n, d = arr.shape
    
    distances = np.linalg.norm(arr - traj[:, np.newaxis, :], axis=2)  # Shape: (T, n)
    summed_distances = np.sum(distances, axis=0)  # Shape: (n,)
    closest_index = np.argmin(summed_distances)
    min_summed_distances = np.min(summed_distances)

    return closest_index, min_summed_distances

def find_most_frequent_trajectory(arr, traj):
    """
    Find the most frequently matched ground truth trajectory and compute the L2 distance.

    Parameters:
    - arr: Ground truth trajectories of shape (T, n, d).
    - traj: Estimated trajectory of shape (T, d).

    Returns:
    - most_frequent_idx: Index of the most frequently matched ground truth trajectory.
    - l2_distance: L2 distance between `traj` and the most frequently matched ground truth trajectory.
    """
    T, n, d = arr.shape

    matched_indices = np.zeros(T, dtype=int)
    for t in range(T):
        distances = np.linalg.norm(arr[t] - traj[t], axis=1)
        matched_indices[t] = np.argmin(distances)

    from collections import Counter
    counter = Counter(matched_indices)
    most_frequent_idx, _ = counter.most_common(1)[0]

    most_frequent_traj = arr[:, most_frequent_idx, :]

    l2_distance = np.linalg.norm(traj - most_frequent_traj)

    return int(most_frequent_idx), l2_distance


def obo_evaluation(ground_data, sampled_data):
    # read data

    columns = ['Jump Probability', 
               'Three-Step Correct Prob', 
               'Five-Step Correct Prob', 
               'Whole Trajectory Correct Prob', 
               'l2 Distance to Best Match (Max)', 
               'l2 Distance to Best Match (Mean)', 
               'Traj Distribution to Uniform (KL-Divergence)']
    
    total_rounds, num_ts, num_particles = sampled_data.shape
    if len(ground_data.shape) == 2:
        ground_data = ground_data.reshape((num_ts, num_particles, 1))
    switch_traj_prob = 0
    three_step_correct = 0
    five_step_correct = 0
    whole_traj_correct = 0
    l2_distance = 0
    l2_max = 0
    traj_distribution = np.zeros(num_particles)
    
    for round in range(total_rounds):
        for ip in range(num_particles):
            traj = sampled_data[round, :, ip]
            single_step_label = (traj[1:] == traj[:-1])
            
            switch_traj_prob += np.mean(1 - single_step_label)
    
            three_step_correct += np.mean(single_step_label[1:] * single_step_label[:-1] == 1)
    
            window_size = 4
            result = np.zeros(len(single_step_label) - window_size + 1, dtype=bool)
            for i in range(len(single_step_label) - window_size + 1):
                result[i] = np.all(single_step_label[i:i + window_size])
            five_step_correct += np.mean(result)
    
            whole_traj_correct += np.all(traj[1:] == traj[0])
    
            max_index, max_length = find_longest_consecutive_trajectory(traj)
            matched_ground_traj = ground_data[:, max_index]
            sampled_traj = ground_data[np.arange(len(traj)), traj]
            l2_distance += np.mean(np.linalg.norm(sampled_traj - matched_ground_traj, axis=1))
            if np.mean(np.linalg.norm(sampled_traj - matched_ground_traj, axis=1)) > l2_max:
                l2_max = np.mean(np.linalg.norm(sampled_traj - matched_ground_traj, axis=1))
    
            traj_distribution[max_index] += 1
    
    switch_traj_prob /= total_rounds * num_particles
    whole_traj_correct /= total_rounds * num_particles
    three_step_correct /= total_rounds * num_particles
    five_step_correct /= total_rounds * num_particles
    l2_distance /= total_rounds * num_particles
    traj_distribution /= total_rounds * num_particles
    # print(np.allclose(np.sum(traj_distribution), 1))
    uniform = np.ones(len(traj_distribution)) / len(traj_distribution)
    kl_to_uniform = entropy(traj_distribution, uniform)

    
    data = [switch_traj_prob, three_step_correct, five_step_correct, whole_traj_correct, l2_max, l2_distance, kl_to_uniform]

    res_dict = dict()
    for col, d in zip(columns, data):
        res_dict[col] = str(d)

    return res_dict

def compute_w2(arr1, arr2):
    weights1 = np.ones(arr1.shape[0]) / arr1.shape[0]
    weights2 = np.ones(arr2.shape[0]) / arr2.shape[0]
    
    cost_matrix = ot.dist(arr1, arr2, metric='sqeuclidean')
    
    w2_distance = np.sqrt(ot.emd2(weights1, weights2, cost_matrix))

    return w2_distance

def compute_w1(arr1, arr2):
    weights1 = np.ones(arr1.shape[0]) / arr1.shape[0]  
    weights2 = np.ones(arr2.shape[0]) / arr2.shape[0]  
    
    cost_matrix = ot.dist(arr1, arr2, metric='euclidean')
    
    w1_distance = ot.emd2(weights1, weights2, cost_matrix)
    
    return w1_distance

def gaussian_kernel_torch(x, y, sigma=1.0):
    #pairwise_dists = torch.sum(x**2, dim=1).unsqueeze(1) + torch.sum(y**2, dim=1) - 2 * torch.mm(x, y.t())
    pairwise_dists = np.sum(x**2, axis=1)[:, np.newaxis] + np.sum(y**2, axis=1) - 2 * np.dot(x, y.T)
    return np.exp(-pairwise_dists / (2 * sigma**2))

def compute_mmd(arr1, arr2, sigma=1.0):
    K_XX = gaussian_kernel_torch(arr1, arr1, sigma)
    K_YY = gaussian_kernel_torch(arr2, arr2, sigma)
    K_XY = gaussian_kernel_torch(arr1, arr2, sigma)

    mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
    return mmd

def mmd_identity_map(arr1, arr2):
    mean_arr1 = np.mean(arr1, axis=0)
    mean_arr2 = np.mean(arr2, axis=0)

    mmd_value = np.linalg.norm(mean_arr1 - mean_arr2, ord=2)

    return mmd_value

    