#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 14:42:51 2023

@author: jose
"""

import torch
from torchvision import datasets, transforms
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# data
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)), 
    ])

train_set = datasets.MNIST('./data', download=True, train=True, transform=transform)
valid_set = datasets.MNIST('./data', download=True, train=False, transform=transform)

conjunto = valid_set
label = 'val'

loader = torch.utils.data.DataLoader(conjunto, batch_size=conjunto.__len__(), shuffle=False)
for _, y in loader:
    break
y = y.numpy()

H = np.load('data/H_{}.npy'.format(label))
Z = np.load('data/Z_{}.npy'.format(label))

# tsne
X_sim = TSNE().fit_transform(H)
plt.scatter(X_sim[:, 0], X_sim[:, 1], c = y)
plt.savefig('output/tsne_H_{}.png'.format(label)), plt.close()

X_sim = TSNE().fit_transform(Z)
plt.scatter(X_sim[:, 0], X_sim[:, 1], c = y)
plt.savefig('output/tsne_Z_{}.png'.format(label)), plt.close()
