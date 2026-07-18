import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load the dataset
dataset = torchvision.datasets.MNIST(
    root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor()
)

# Feed it into a dataloader (batch size 8 here just for demo)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

if __name__ == '__main__':
    # View some examples
    x, y = next(iter(train_dataloader))
    print('Input shape:', x.shape)
    print('Labels:', y)
    plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
    plt.show()
