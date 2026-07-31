import numpy as np
import torch
import torch.nn as nn
from get_data import dataset
from define_model import DiTFullModel
from torch.utils.data import DataLoader, Subset
from config import config
import wandb
import modal


image = modal.Image.debian_slim(
    python_version="3.12"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "diffusers", "tqdm", "wandb"
).add_local_python_source(
    "define_model", "get_data", "config"
)
app = modal.App("mastering-pytorch-my_dit")


def train_procedure():
    """
    device = pick cuda, else mps, else cpu
    dataloader = DataLoader over the dataset
    net, optimizer (Adam), loss_fn (MSELoss)
    for epoch in range(n_epochs):
      for x, y in dataloader:
        x = x * 2 - 1  # normalize to [-1, 1]
        t = Uniform(0, 1), ONE INDEPENDENT VALUE PER SAMPLE, shape (bs,) # not one shared scalar for the batch
        reshaped_t = t reshaped to (bs, 1, 1, 1) # for broadcasting against the image
        x_0 = standard Gaussian noise, same shape as x
        x_1 = x  # the real image
        x_t = (1 - reshaped_t) * x_0 + reshaped_t * x_1 # straight-line interpolation
        target = x_1 - x_0  # velocity
        pred = net(x_t, t, y)  # t stays (bs,) here, NOT reshaped_t
        loss = MSE(pred, target)
        backward + optimizer step
      log epoch average loss
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    wandb.init(
        project='my_dit',
        name=f'hdn={config.hidden_size}|lyrs={config.num_layers}|epchs={config.n_epochs}',
        config=config
    )
    # subset = Subset(dataset, indices=range(1000))
    subset = dataset
    dataloader = DataLoader(subset, batch_size=config.batch_size, shuffle=True)
    net: DiTFullModel = DiTFullModel(
        side=config.side_size,
        path_side=config.patch_size,
        head_dim=config.hidden_size // config.num_heads,
        hidden_size=config.hidden_size,
        num_classes=config.num_classes,
        class_emb_size=config.class_emb_size,
        time_emb_size=config.time_emb_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        channels=config.channels
    ).to(device)
    print(f'num params: {sum(p.numel() for p in net.parameters())}')
    print(f'hidden_size={config.hidden_size} | num_layers={config.num_layers} | n_epochs={config.n_epochs}')
    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_fn: nn.MSELoss = nn.MSELoss()

    losses = []
    for epoch in range(config.n_epochs):
        epoch_losses = []
        for data_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            bs, ch, width, height = x.shape
            x = x * 2 - 1
            t: torch.Tensor = torch.rand((bs,)).to(device)
            reshaped_t = t.reshape((bs, 1, 1, 1))
            x_0 = torch.randn_like(x).to(device)
            x_1 = x
            x_t = (1 - reshaped_t) * x_0 + reshaped_t * x_1  # straight-line interpolation
            target = x_1 - x_0
            pred = net(x_t, t, y)
            loss = loss_fn(pred, target)
            net.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            wandb.log({'loss': loss.item()})
            print(f'\r[{epoch} | {data_idx}] loss={loss.item()}', end='')
            yield {'type': 'msg', 'loss': loss.item()}
        avr_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        print(f'\n[{epoch=}] avr loss={avr_epoch_loss: .3f}')
        wandb.log({'avr_epoch_loss': avr_epoch_loss})
    yield {'type': 'final', 'state_dict': {k: v.cpu() for k, v in net.state_dict().items()}}


@app.function(image=image, gpu="A10G", timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")])
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