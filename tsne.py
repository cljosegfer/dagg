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

conjunto = train_set
label = 'train'

loader = torch.utils.data.DataLoader(conjunto, batch_size=conjunto.__len__(), shuffle=False)
for _, y in loader:
    break
y = y.numpy()

H = np.load('data/H_{}.npy'.format(label))
Z = np.load('data/Z_{}.npy'.format(label))

# tsne
H_tsne = TSNE().fit_transform(H)
np.save('data/H_tsne_{}.npy'.format(label), H_tsne)
plt.scatter(H_tsne[:, 0], H_tsne[:, 1], c = y)
plt.savefig('output/tsne_H_{}.png'.format(label)), plt.close()

Z_tsne = TSNE().fit_transform(Z)
np.save('data/Z_tsne_{}.npy'.format(label), Z_tsne)
plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c = y)
plt.savefig('output/tsne_Z_{}.png'.format(label)), plt.close()
