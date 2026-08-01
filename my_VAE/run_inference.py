from typing import Any
import matplotlib.pyplot as plt
import torch
from get_data import dataset
from config import config
from define_model import VAEFullModel


def plot_image(image, pause_time=0.01):
    plt.cla()
    plt.imshow(image)
    plt.pause(pause_time)


def run_inference():
    """
    # 1. Prior sampling — pure generation, single decoder pass, no iterative loop needed (unlike diffusion/flow matching)
    z = standard Gaussian noise, shape (num_samples, latent_dim)
    images = decoder(z)

    # 2. Reconstruction check — encode real images, decode back, compare to originals
    mu, logvar = encoder(real_images)
    reconstructed = decoder(mu)              # use mu directly, not reparameterize() — see gotcha

    # 3. Latent interpolation — the classic "prove the latent space is smooth" demo
    mu_a, _ = encoder(image_a)
    mu_b, _ = encoder(image_b)
    for step in range(num_interpolation_steps):
      alpha = step / (num_interpolation_steps - 1)
      z_interp = (1 - alpha) * mu_a + alpha * mu_b
      images.append(decoder(z_interp))
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bs = 10
    net = VAEFullModel(
        channels=config.channels,
        side_size=config.side_size,
        base_channels=config.base_channels,
        latent_dim=config.latent_dim
    ).to(device)
    net.load_state_dict(state_dict=torch.load('saved_model.pt', map_location=device))
    net.eval()
    print(f'num params: {sum(p.numel() for p in net.parameters())}')
    with torch.no_grad():
        z = torch.randn((bs, config.latent_dim))
        images = net.decoder(z)
        images = (images + 1) / 2
        images = images.reshape((5, 2, 3, 32, 32)).permute(0, 3, 1, 4, 2).reshape((5 * 32, 2 * 32, 3)).numpy()
        plot_image(images)
        plt.show()
    yield {'type': 'img', 'images': images}

def main():
    x_list = []
    for chunk in run_inference():
        continue


if __name__ == '__main__':
    main()