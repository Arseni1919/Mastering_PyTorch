import torch
import torch.nn as nn
from get_data import dataset
from define_vae import Encoder, Decoder
from define_dit import DiTFullModel
from torch.utils.data import DataLoader
from config import config
import wandb
import modal


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
    "define_vae", "get_data", "config", "define_dit"
)
app = modal.App("mastering-pytorch")

vae_volume = modal.Volume.from_name("vae-checkpoints")


def train_procedure():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    wandb.init(
        project='my_dit',
        config=config,
        # mode='disabled'
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
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
    encoder: Encoder = Encoder(
        channels=config.channels,
        vae_base_channels=config.vae_base_channels,
        vae_blocks_per_stage=config.vae_blocks_per_stage,
        vae_latent_channels=config.vae_latent_channels,
        vae_patch_size=config.vae_patch_size,
        num_downsample_stages=config.num_downsample_stages,
        num_temporal_downsample_stages=config.num_temporal_downsample_stages
    ).to(device)
    encoder_state_dict = torch.load(f'/root/vae_checkpoints/{config.vae_model_name}_encoder.pt', map_location=device)
    encoder.load_state_dict(state_dict=encoder_state_dict)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f'encoder num params: {sum(p.numel() for p in net.parameters())}')
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn: nn.MSELoss = nn.MSELoss(reduction='mean')
    n_datapoints = len(dataloader)

    losses = []
    for epoch in range(config.n_epochs_dit):
        epoch_losses = []
        for data_idx, (x, y) in enumerate(dataloader):
            x: torch.Tensor = x.to(device); y = y.to(device)
            bs, ch, ts, h_, w_ = x.shape
            x = x * 2 - 1
            with torch.no_grad():
                mu, logvar = encoder(x)
            z_1 = mu
            t = torch.rand((bs,))
            t = t.to(device)
            reshaped_t = t.reshape((bs, 1, 1, 1, 1))
            z_0 = torch.randn_like(z_1)
            z_t = (1 - reshaped_t) * z_0 + reshaped_t * z_1
            target = z_1 - z_0
            pred = net(z_t, t, y)
            loss = loss_fn(pred, target)
            net.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            wandb.log({'loss': loss.item()})
            print(f'\r[{epoch}/{config.n_epochs_dit} | {data_idx}/{n_datapoints}] loss={loss.item()}', end='')
            yield {'type': 'msg', 'loss': loss.item()}
        avr_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        print(f'\n[{epoch=}] avr loss={avr_epoch_loss: .3f}')
        wandb.log({'avr_epoch_loss': avr_epoch_loss})
    yield {
        'type': 'final',
        'state_dict_dit': {k: v.cpu() for k, v in net.state_dict().items()},
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/root/KTH_Action": kth_volume, "/root/vae_checkpoints": vae_volume},
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
            state_dict_encoder = chunk["state_dict_dit"]
            torch.save(state_dict_encoder, f'state_dict_dit.pt')
    print(f'--- finished ---')


def main():
    for chunk in train_procedure():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict_encoder = chunk["state_dict_dit"]
            torch.save(state_dict_encoder, f'state_dict_dit.pt')
    print(f'--- finished ---')


if __name__ == '__main__':
    main()