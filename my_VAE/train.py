import torch
import torch.nn as nn
from get_data import dataset
from define_model import VAEFullModel, kl_divergence
from config import config
import wandb
import modal

image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb"
).env(
    {"CIFAR_DATA_ROOT": "/data/cifar"}
).add_local_python_source(
    "define_model", "get_data", "config"
)
app = modal.App("mastering-pytorch-my_vae")
cifar_volume = modal.Volume.from_name("cifar10-data", create_if_missing=True)


def train_procedure():
    """
    device = pick cuda, else mps, else cpu
    dataloader = DataLoader over the dataset
    net, optimizer (Adam), reconstruction loss (MSE)

    for epoch in range(n_epochs):
      for x, _ in dataloader:                # label unused — unconditional
        x = x.to(device) * 2 - 1             # normalize to [-1, 1]
        x_hat, mu, logvar = net(x)
        recon_loss = MSE(x_hat, x)
        kl_loss = kl_divergence(mu, logvar)
        loss = recon_loss + kl_weight * kl_loss
        zero_grad + backward + optimizer step
      log recon_loss and kl_loss SEPARATELY, not just the combined loss
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=config.batch_size)
    net = VAEFullModel(
        channels=config.channels,
        side_size=config.side_size,
        base_channels=config.base_channels,
        latent_dim=config.latent_dim
    ).to(device)
    print(f'num params: {sum(p.numel() for p in net.parameters())}\n')
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss(reduction='sum')
    wandb.init(project='my_vae', config=config)

    for epoch in range(config.n_epochs):
        epoch_losses = []
        for data_idx, (x, _) in enumerate(dataloader):
            x = x.to(device) * 2 - 1
            x_hat, mu, logvar = net(x)
            recon_loss = loss_fn(x_hat, x) / x.shape[0]
            kl_loss = kl_divergence(mu, logvar)
            loss = recon_loss + config.kl_weight * kl_loss
            net.zero_grad()
            loss.backward()
            optim.step()

            wandb.log({'loss': loss.item(), 'recon_loss': recon_loss.item(), 'kl_loss': kl_loss.item()})
            epoch_losses.append(loss.item())
            print(f'\r[{epoch} | {data_idx}] {loss.item()=: .3f}, {recon_loss.item()=: .3f}, {kl_loss.item()=: .3f}', end='')
            yield {'type': 'msg', 'loss': loss.item()}
        avr_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        wandb.log({'avr_epoch_loss': avr_epoch_loss})
    yield {'type': 'final', 'state_dict': {k: v.cpu() for k, v in net.state_dict().items()}}


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/data": cifar_volume}
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
            state_dict = chunk["state_dict"]
            torch.save(state_dict, 'saved_model.pt')
    print(f'--- finished ---')



def main():
    for chunk in train_procedure():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict = chunk["state_dict"]
            torch.save(state_dict, 'saved_model.pt')
    print(f'--- finished ---')

if __name__ == '__main__':
    main()