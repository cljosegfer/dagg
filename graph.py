#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 12:32:05 2023

@author: jose
"""

import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
from itertools import combinations
from datetime import datetime

from tqdm import tqdm

# Function to compute the Gabriel Graph using PyTorch
def GabrielGraph(data):
    # This function computes the Gabriel Graph of an input data.
    #
    # It first calculates the squared distance array(SDA) containing the squared euclidean distance
    # between the data points.
    #
    # Then it returns the graph, or array of adjacency.

    data = torch.tensor(data).cuda()
    n = data.size(0)
    # fourth_power_distance_array = torch.zeros((n, n), dtype=torch.float64).cuda()

    # for i in tqdm(range(n)):
    #     fourth_power_distance_array[:, i] = torch.sum((data - data[i]).pow(2), dim=1).pow(2)  # vectorized fourth power distance of col(i)
    #     fourth_power_distance_array[i, i] = float('inf')  # distance between same point is infinity (convention used in this code)

    # array_of_adjacency = torch.zeros((n, n), dtype=torch.int32).cuda()
    min_sum_of_distances = torch.empty(n, dtype=torch.float64).cuda()
    list_of_adjacency = []

    for i in tqdm(range(n - 1)):  # No need to iterate the last row.
        fourth_power_distance_veci = torch.sum((data - data[i]).pow(2), dim=1).pow(2)
        fourth_power_distance_veci[i] = float('inf')
        for j in range(i + 1, n):  # No need to iterate over j <= i
            fourth_power_distance_vecj = torch.sum((data - data[j]).pow(2), dim=1).pow(2)
            fourth_power_distance_vecj[j] = float('inf')
            # min_sum_of_distances = torch.min(fourth_power_distance_array[i, :] + fourth_power_distance_array[j, :])
            min_sum_of_distances = torch.min(fourth_power_distance_veci + fourth_power_distance_vecj)
            # if fourth_power_distance_array[i, j] <= min_sum_of_distances:
            if fourth_power_distance_veci[j] <= min_sum_of_distances:
                # if the sum of the minimum distances between other points to i and j isn't
                # less or equal to the distance between i and j, then (i,j) is an edge that belongs to the graph.
                # array_of_adjacency[i, j] = 1
                # array_of_adjacency[j, i] = 1
                list_of_adjacency.append([i, j])

    # return array_of_adjacency.cpu().numpy()
    # edges = torch.nonzero(array_of_adjacency, as_tuple=False)
    # return edges.cpu().numpy()
    return np.array(list_of_adjacency)

# data
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)), 
    ])

train_set = datasets.MNIST('./data', download=True, train=True, transform=transform)
valid_set = datasets.MNIST('./data', download=True, train=False, transform=transform)

conjunto = train_set
label = 'train'

loader = torch.utils.data.DataLoader(conjunto, batch_size=conjunto.__len__(), shuffle=False)
for _, y in loader:
    break
y = y.numpy()

H = np.load('data/H_{}.npy'.format(label))
Z = np.load('data/Z_{}.npy'.format(label))

# gg
# sz = 1000
# print(sz)
# X = Z[np.random.choice(len(Z), size = sz), :]
# X = torch.tensor(X)
# X = torch.tensor(H)

Z_gg = GabrielGraph(Z)
np.save('data/Z_gg_{}.npy'.format(label), Z_gg)
H_gg = GabrielGraph(H)
np.save('data/H_gg_{}.npy'.format(label), H_gg)
