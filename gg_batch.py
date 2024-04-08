
import torch
import numpy as np

from tqdm import tqdm

from gabrielgraph.gg import GG

H = np.load('data/H_train.npy')
batch_size = 100
sz = 1000
print(sz, batch_size)
idx = np.random.choice(len(H), size = sz)
X = H[idx, :]
# X = H

ggclass = GG()
adj = ggclass.torch(X)

adj_batch = ggclass.torch_batch(X, btsz = batch_size, tol = 1e-6)

label, counts = np.unique(adj.cpu() == adj_batch.cpu(), return_counts = True)
print(label, counts / sum(counts))
