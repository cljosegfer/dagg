#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 13:30:55 2023

@author: jose
"""

import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from tqdm import tqdm

from util import TwoCropTransform
from model import Encoder, SupCon, SupConLoss

# param
head_type = 'mlp'
num_epochs = 100

# data
contrastive_transform = transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                                   ])
train_transform = transforms.Compose([
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                                   ])
valid_transform = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,)),
                                   ])

contrastive_set = datasets.MNIST('./data', download=True, train=True, transform=TwoCropTransform(contrastive_transform))
train_set = datasets.MNIST('./data', download=True, train=True, transform=train_transform)
valid_set = datasets.MNIST('./data', download=True, train=False, transform=valid_transform)

contrastive_loader = torch.utils.data.DataLoader(contrastive_set, batch_size=64, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64, shuffle=True)

# model
encoder = Encoder()
model = SupCon(encoder, head=head_type, feat_dim=128)
criterion = SupConLoss(temperature=0.07)
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

log = []
for epoch in tqdm(range(1, num_epochs+1)):
    model.train()
    train_loss = 0
    for batch_idx, (data,labels) in enumerate(contrastive_loader):
        data = torch.cat([data[0], data[1]], dim=0)
        if torch.cuda.is_available():
            data,labels = data.cuda(), labels.cuda()
        data, labels = torch.autograd.Variable(data,False), torch.autograd.Variable(labels)
        bsz = labels.shape[0]
        features = model(data)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    scheduler.step()
    print(train_loss / len(contrastive_loader))
    log.append(train_loss / len(contrastive_loader))

plt.plot(range(1,len(log)+1),log, color='b', label = 'loss')
plt.legend(), plt.ylabel('loss'), plt.xlabel('epochs'), plt.title('Loss'), plt.savefig('output/plot2.png'), plt.close()
