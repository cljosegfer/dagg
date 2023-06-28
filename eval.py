
import torch
from torchvision import datasets, transforms
import numpy as np

from tqdm import tqdm

from model import Encoder, SupCon, SupConLoss

# model
encoder = Encoder()
model = SupCon(encoder, head='mlp', feat_dim=128)
model = model.cuda()

# load
ckpt = torch.load('output/model.pth.tar')
state_dict = ckpt['model']
model.load_state_dict(state_dict)

# data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])

train_set = datasets.MNIST('./data', download=True, train=True, transform=transform)
valid_set = datasets.MNIST('./data', download=True, train=False, transform=transform)

label = 'val'
conjunto = valid_set

bsz = 100
loader = torch.utils.data.DataLoader(conjunto, batch_size=bsz, shuffle=False)

# eval
H = np.zeros(shape = (conjunto.__len__(), 2048))
Z = np.zeros(shape = (conjunto.__len__(), 128))

model.eval()
with torch.no_grad():
    for batch_idx, (data,labels) in tqdm(enumerate(loader)):
        data = data.cuda()
        h = model.encoder(data)
        z = model.forward(data)
        H[batch_idx*bsz:batch_idx*bsz+bsz, :] = h.cpu().detach().numpy()
        Z[batch_idx*bsz:batch_idx*bsz+bsz, :] = z.cpu().detach().numpy()

# export
np.save('data/H_{}.npy'.format(label), H)
np.save('data/Z_{}.npy'.format(label), Z)
