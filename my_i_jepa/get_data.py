import matplotlib.pyplot as plt
import torchvision
import random


dataset = torchvision.datasets.MNIST(
    root='mnist',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)


def main():
    x, y = dataset[random.randint(0, len(dataset) - 1)]
    x = x.permute(2, 1, 0).transpose(0, 1).numpy()
    plt.imshow(x)
    plt.show()



if __name__ == '__main__':
    main()