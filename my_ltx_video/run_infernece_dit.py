import matplotlib.pyplot as plt
import torch
from config import config
from define_vae import Decoder
from define_dit import DiTFullModel
from tqdm import tqdm
from get_data import dataset


def plot_image(image, timestamp, model_config, pause_time=0.01):
    plt.cla()
    plt.imshow(image)
    plt.title(
        f'timestamp: {timestamp + 1}\n'
        f'hidden_size={model_config.hidden_size} | '
        f'num_layers={model_config.num_layers} | '
        f'n_epochs={model_config.n_epochs_dit}'
    )
    plt.pause(pause_time)


def run_inference():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder: Decoder = Decoder(
        channels=config.channels,
        vae_base_channels=config.vae_base_channels,
        vae_blocks_per_stage=config.vae_blocks_per_stage,
        vae_latent_channels=config.vae_latent_channels,
        vae_patch_size=config.vae_patch_size,
        num_downsample_stages=config.num_downsample_stages,
        num_temporal_downsample_stages=config.num_temporal_downsample_stages
    ).to(device)
    decoder.load_state_dict(state_dict=torch.load(f'{config.vae_model_name}_decoder.pt', map_location=device))
    decoder.eval()
    T = config.num_frames // (2 ** config.num_temporal_downsample_stages)
    H = W = config.side_size // (config.vae_patch_size * (2 ** (config.num_downsample_stages - 1)))
    net: DiTFullModel = DiTFullModel(
        vae_latent_channels=config.vae_latent_channels,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
        num_classes=config.num_classes,
        class_emb_size=config.class_emb_size,
        T=T, H=H, W=W
    ).to(device)
    net.load_state_dict(state_dict=torch.load(f'state_dict_dit.pt', map_location=device))
    net.eval()
    print(f'num params: {sum(p.numel() for p in net.parameters())}')
    T = config.num_frames // (2 ** config.num_temporal_downsample_stages)
    H = W = config.side_size // (config.vae_patch_size * (2 ** (config.num_downsample_stages - 1)))

    for _ in range(10):
        z = torch.randn(1, config.vae_latent_channels, T, H, W).to(device)
        t = torch.zeros(1).to(device)
        class_labels = torch.randint(0, config.num_classes, (1,)).to(device)
        first_class_index = int(class_labels[0].item())
        print(f'{first_class_index=}')
        step_size = 1 / config.num_inference_steps
        for _ in tqdm(range(config.num_inference_steps)):
            with torch.no_grad():
                pred = net(z, t, class_labels)
            z += step_size * pred
            t += step_size

        show_x = decoder(z).detach()
        print('Start to show...')
        for i in range(config.num_frames):
            plt.cla()
            frame = show_x.squeeze(0)[0, i, :, :].unsqueeze(2)
            plt.imshow(frame)
            plt.title(f'frame {i} | class = {dataset.classes[first_class_index]}')
            plt.pause(0.05)
    plt.show()



def main():
    run_inference()


if __name__ == '__main__':
    main()