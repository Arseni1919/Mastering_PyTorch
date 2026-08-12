import torch
from get_data import dataset
from define_model import IJEPAModel
from torch.utils.data import DataLoader
from config import config
import wandb
import modal

image = modal.Image.debian_slim(
    python_version="3.12"
).apt_install(
    "ffmpeg"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "tqdm", "wandb", "transformers",
).add_local_python_source(
    "define_model", "get_data", "config"
)
app = modal.App("mastering-pytorch")


def train_procedure():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    wandb.init(
        project='i-jepa',
        config=config,
        # mode='disabled'
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = IJEPAModel(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        embed_dim=config.embed_size,
        encoder_depth=config.encoder_depth,
        encoder_num_heads=config.encoder_num_heads,
        predictor_embed_dim=config.predictor_embed_dim,
        predictor_depth=config.predictor_depth,
        predictor_num_heads=config.predictor_num_heads,
        sigreg_num_slices=config.sigreg_num_slices,
        sigreg_lambda=config.sigreg_lambda,
        M=4,
        target_scale_range=config.target_scale_range,
        target_aspect_ratio_range=config.target_aspect_ratio_range,
        context_scale_range=config.context_scale_range
    ).to(device)
    print(f'encoder num params: {sum(p.numel() for p in model.parameters())}')
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate
    )
    n_datapoints = len(dataloader)

    losses = []
    for epoch in range(config.epochs):
        epoch_losses = []
        for data_idx, (x, y) in enumerate(dataloader):
            x: torch.Tensor = x.to(device)
            loss, prediction_loss, sigreg = model(x)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            wandb.log({
                'loss': loss.item(),
                'prediction_loss': prediction_loss.item(),
                'sigreg': sigreg.item(),
            })
            print(f'\r[{epoch}/{config.epochs} | {data_idx}/{n_datapoints}] loss={loss.item()}', end='')
            if data_idx % 500 == 0 and data_idx != 0:
                yield {
                    'type': 'final',
                    'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                }
            else:
                yield {'type': 'msg', 'loss': loss.item()}
        avr_epoch_loss = sum(epoch_losses)/len(epoch_losses)
        print(f'\n[{epoch=}] avr loss={avr_epoch_loss: .3f}')
        wandb.log({'avr_epoch_loss': avr_epoch_loss})
    yield {
        'type': 'final',
        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
    }


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
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
            state_dict = chunk["state_dict"]
            torch.save(state_dict, f'state_dict.pt')
    print(f'--- finished ---')


def main():
    for chunk in train_procedure():
        if chunk['type'] == 'msg':
            continue
        if chunk['type'] == 'final':
            state_dict = chunk["state_dict"]
            torch.save(state_dict, f'state_dict.pt')
    print(f'--- finished ---')


if __name__ == '__main__':
    main()