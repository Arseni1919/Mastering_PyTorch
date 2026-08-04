import numpy as np
import torch
import torch.nn as nn
from get_data import dataset
from define_vae import Encoder, Decoder, reparameterize, kl_divergence
from torch.utils.data import DataLoader
from config import config
import wandb
import modal
import lpips

kth_volume = modal.Volume.from_name("kth-action-data")

image = modal.Image.debian_slim(
    python_version="3.12"
).apt_install(
    "ffmpeg"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb", "torchcodec", "lpips"
).env(
    {"KTH_DATA_ROOT": "/root/KTH_Action/KTH_Action"}
).add_local_python_source(
    "define_vae", "get_data", "config"
)
app = modal.App("mastering-pytorch")


def train_procedure():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    wandb.init(
        project='my_dit',
        config=config,
        # mode='disabled'
    )
    subset = dataset
    dataloader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
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
    print(f'encoder num params: {sum(p.numel() for p in encoder.parameters())}')
    print(f'decoder num params: {sum(p.numel() for p in decoder.parameters())}')
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=config.learning_rate)
    n_datapoints = len(dataloader)
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)  # or net='alex' for lighter/faster
    for p in lpips_loss_fn.parameters():
        p.requires_grad = False  # freeze it, it's just a loss metric

    losses = []
    for epoch in range(config.n_epochs_vae):
        epoch_losses = []
        for data_idx, (x, y) in enumerate(dataloader):
            x: torch.Tensor = x.to(device); y = y.to(device)
            bs, ch, ts, h_, w_ = x.shape
            x = x * 2 - 1
            mu, logvar = encoder(x)
            z = reparameterize(mu, logvar)
            x_hat = decoder(z)
            recon_loss = torch.abs(x_hat - x).mean()
            reg_loss = kl_divergence(mu, logvar)
            lpips_x = x.permute(0,2,1,3,4).reshape((bs*ts,ch,h_,w_)).repeat(1,3,1,1)
            lpips_x_hat = x_hat.permute(0,2,1,3,4).reshape((bs*ts,ch,h_,w_)).repeat(1,3,1,1)
            perceptual = lpips_loss_fn(lpips_x_hat, lpips_x).mean()
            loss = recon_loss + config.vae_kl_weight * reg_loss + 1.0 * perceptual
            encoder.zero_grad(); decoder.zero_grad()
            loss.backward()
            encoder_optimizer.step(); decoder_optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            wandb.log({'loss': loss.item()})
            wandb.log({'recon_loss': recon_loss.item()})
            wandb.log({'reg_loss': reg_loss.item()})
            print(f'\r[{epoch}/{config.n_epochs_vae} | {data_idx}/{n_datapoints}] loss={loss.item()}', end='')
            yield {'type': 'msg', 'loss': loss.item()}
        avr_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        print(f'\n[{epoch=}] avr loss={avr_epoch_loss: .3f}')
        wandb.log({'avr_epoch_loss': avr_epoch_loss})
    yield {
        'type': 'final',
        'state_dict_encoder': {k: v.cpu() for k, v in encoder.state_dict().items()},
        'state_dict_decoder': {k: v.cpu() for k, v in decoder.state_dict().items()}
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/root/KTH_Action": kth_volume},
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def modal_train():
    for chunk in train_procedure():
        yield chunk

@app.local_entrypoint()
def modal_main():
    for chunk in modal_train.remote_gen():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict_encoder = chunk["state_dict_encoder"]
            torch.save(state_dict_encoder, f'{config.vae_model_name}_encoder.pt')
            state_dict_decoder = chunk["state_dict_decoder"]
            torch.save(state_dict_decoder, f'{config.vae_model_name}_decoder.pt')
    print(f'--- finished ---')


def main():
    for chunk in train_procedure():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict_encoder = chunk["state_dict_encoder"]
            torch.save(state_dict_encoder, f'{config.vae_model_name}_encoder.pt')
            state_dict_decoder = chunk["state_dict_decoder"]
            torch.save(state_dict_decoder, f'{config.vae_model_name}_decoder.pt')
    print(f'--- finished ---')

if __name__ == '__main__':
    main()