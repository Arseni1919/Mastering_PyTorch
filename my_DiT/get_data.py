import torchvision
import torch


dataset = torchvision.datasets.MNIST(
    root='mnist', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.Pad(2),
        torchvision.transforms.ToTensor()
    ])
)


if __name__ == '__main__':
    x, y = dataset[0]
    print(x.shape)
    print(y)