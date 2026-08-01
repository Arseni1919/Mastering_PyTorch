import os
import torch
import torchvision
import matplotlib.pyplot as plt

root = os.environ.get('CIFAR_DATA_ROOT', 'cifar')
dataset = torchvision.datasets.CIFAR10(
    root,
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

if __name__ == '__main__':
    x, y = dataset[0]
    print(x.shape)
    print(y)
    pics = [dataset[i][0].unsqueeze(0) for i in range(100)]
    pics = torch.cat(pics)
    bs, ch, h, w = pics.shape
    pics = pics.reshape(10, 10, ch, h, w)
    pics = pics.permute(0, 3, 1, 4, 2).reshape(10*h, 10*w, ch)
    plt.imshow(pics.numpy())
    plt.show()
