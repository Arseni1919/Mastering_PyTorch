import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = torchvision.datasets.MNIST(root='mnist', train=True, download=True, transform=transforms.Compose([
  transforms.ToTensor()
]))
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


if __name__ == '__main__':
  x, y = next(iter(dataloader))
  print(x.shape, y.shape)
  plt.imshow(torchvision.utils.make_grid(x)[0], cmap='gray')
  plt.show()


