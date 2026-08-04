import matplotlib.pyplot as plt
import torch
from get_data import dataset
from config import config
from define_vae import Encoder, Decoder, reparameterize
import random


def plot_image(image, timestamp, model_config, pause_time=0.01):
    plt.cla()
    plt.imshow(image)
    plt.title(
        f'timestamp: {timestamp + 1}\n'
        f'hidden_size={model_config.hidden_size} | '
        f'num_layers={model_config.num_layers} | '
        f'n_epochs={model_config.n_epochs}'
    )
    plt.pause(pause_time)


def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder: Encoder = Encoder(
        channels=config.channels,
        vae_base_channels=config.vae_base_channels,
        vae_blocks_per_stage=config.vae_blocks_per_stage,
        vae_latent_channels=config.vae_latent_channels,
        vae_patch_size=config.vae_patch_size,
        num_downsample_stages=config.num_downsample_stages,
        num_temporal_downsample_stages=config.num_temporal_downsample_stages
    ).to(device)
    decoder: Decoder = Decoder(
        channels=config.channels,
        vae_base_channels=config.vae_base_channels,
        vae_blocks_per_stage=config.vae_blocks_per_stage,
        vae_latent_channels=config.vae_latent_channels,
        vae_patch_size=config.vae_patch_size,
        num_downsample_stages=config.num_downsample_stages,
        num_temporal_downsample_stages=config.num_temporal_downsample_stages
    ).to(device)
    encoder.load_state_dict(state_dict=torch.load(f'{config.vae_model_name}_encoder.pt', map_location=device))
    decoder.load_state_dict(state_dict=torch.load(f'{config.vae_model_name}_decoder.pt', map_location=device))
    encoder.eval()
    decoder.eval()
    fig, ax = plt.subplots(1, 2)
    for v in range(10):
        x, y = dataset[random.randint(0, len(dataset)-1)]
        x: torch.Tensor = x * 2 - 1
        x = x.unsqueeze(0)
        mu, logvar = encoder(x)
        z = reparameterize(mu, logvar)
        x_hat = decoder(z)
        x_hat = (x_hat + 1) / 2
        x_hat = x_hat.squeeze(0).detach()
        for i in range(config.num_frames):
            ax[0].cla()
            frame = x.squeeze(0)[0, i, :, :].unsqueeze(2)
            ax[0].imshow(frame)
            ax[0].set_title(f'[{v}] frame {i} - ORIG')
            # ---
            ax[1].cla()
            frame = x_hat[0,i,:,:].unsqueeze(2)
            ax[1].imshow(frame)
            ax[1].set_title(f'[{v}] frame {i} - PRED')
            plt.pause(0.01)
    plt.show()



def main():
    run_inference()


if __name__ == '__main__':
    main()