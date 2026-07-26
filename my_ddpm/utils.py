import matplotlib.pyplot as plt


def plot_losses(losses: list):
    plt.plot(losses)
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()


def plot_image(image, pause_time=0.01):
    plt.imshow(image)
    plt.pause(pause_time)
