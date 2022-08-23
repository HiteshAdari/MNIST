import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()

torch.cuda.init()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

dataset = torchvision.datasets.MNIST(root='data/', transform=torchvision.transforms.ToTensor())
'''
#Visualizing image
image, label = dataset[0]
print('image.shape:', image.shape)
plt.imshow(image.permute(1, 2, 0), cmap='gray')
print('Label:', label)
plt.show()
'''

val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator(device='cuda'))
# print(len(train_ds), len(val_ds))

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size * 2, num_workers=4, pin_memory=True)

for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    plt.show()
    break
