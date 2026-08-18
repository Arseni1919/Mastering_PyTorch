import torchvision
import os


# dataset = torchvision.datasets.MNIST(
#     root='mnist',
#     train=True,
#     download=True,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor()
# ]))

root = os.environ.get('CIFAR_DATA_ROOT', 'cifar')
dataset = torchvision.datasets.CIFAR10(
    root,
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)


def main():
    x, y = dataset[0]
    print(x.shape)


if __name__ == '__main__':
    main()