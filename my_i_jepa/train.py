import torch
from get_data import dataset
from define_model import IJEPAModel
from torch.utils.data import DataLoader
import torch.optim as optim
from config import config
import wandb
import modal
import math


stl10_volume = modal.Volume.from_name("stl10-data", create_if_missing=True)
# ckpts_vol = modal.Volume.from_name("ckpts", create_if_missing=True)


image = modal.Image.debian_slim(
    python_version="3.12"
).apt_install(
    "ffmpeg"
).uv_pip_install(
    "torch==2.13.0", "torchvision", "matplotlib", "tqdm", "wandb", "transformers",
).env(
    {
        "STL10_DATA_ROOT": "/root/stl10",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
     }
).add_local_python_source(
    "define_model", "get_data", "config"
)
app = modal.App("mastering-pytorch")


dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def cosine_lr_with_warmup(step: int) -> float:
    """Linear warmup then cosine decay; returns LR multiplier."""
    OPT_STEPS_PER_EPOCH = len(dataloader) // config.accum_steps
    WARMUP_OPT_STEPS = config.warmup_epochs * OPT_STEPS_PER_EPOCH
    TOTAL_OPT_STEPS = config.epochs * OPT_STEPS_PER_EPOCH
    if step < WARMUP_OPT_STEPS:
        return (step + 1) / WARMUP_OPT_STEPS
    progress = (step - WARMUP_OPT_STEPS) / max(TOTAL_OPT_STEPS - WARMUP_OPT_STEPS, 1)
    return config.lr_min / config.lr + 0.5 * (1.0 - config.lr_min / config.lr) * (1.0 + math.cos(math.pi * progress))


def train_procedure():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'--- DEVICE: {device} ---')
    if device == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    wandb.init(
        project='i-jepa',
        config=config,
        # mode='disabled'
    )
    model = IJEPAModel(
        image_size=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        encoder_embed_dim=config.encoder_embed_dim,
        encoder_depth=config.encoder_depth,
        encoder_num_heads=config.encoder_num_heads,
        predictor_embed_dim=config.predictor_embed_dim,
        predictor_depth=config.predictor_depth,
        predictor_num_heads=config.predictor_num_heads,
        sigreg_num_slices=config.sigreg_num_slices,
        sigreg_lambda=config.sigreg_lambda,
        num_target_blocks=config.num_target_blocks,
        target_scale_range=config.target_scale_range,
        target_aspect_ratio_range=config.target_aspect_ratio_range,
        context_scale_range=config.context_scale_range,
        ema_start=config.ema_start,
        ema_end=config.ema_end
    ).to(device)
    print(f'encoder num params: {sum(p.numel() for p in model.parameters()):_}')
    # optimizer = torch.optim.Adam(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=config.lr
    # )
    optimizer = optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()),
        lr=config.lr, betas=(0.9, 0.95), weight_decay=config.wd_start,
    )
    global_opt_step = 0
    OPT_STEPS_PER_EPOCH = len(dataloader) // config.accum_steps
    TOTAL_OPT_STEPS = config.epochs * OPT_STEPS_PER_EPOCH
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_lr_with_warmup)
    n_datapoints = len(dataloader)

    losses = []
    for epoch in range(config.epochs):
        epoch_losses = []
        for data_idx, (x, y) in enumerate(dataloader):
            x: torch.Tensor = x.to(device)
            loss, prediction_loss, sigreg = model(x)

            (loss / config.accum_steps).backward()
            if (data_idx + 1) % config.accum_steps == 0:

                # linear weight decay schedule: WD_START → WD_END
                cur_wd = config.wd_start + (config.wd_end - config.wd_start) * (
                        global_opt_step / max(TOTAL_OPT_STEPS - 1, 1))
                for pg in optimizer.param_groups:
                    pg['weight_decay'] = cur_wd

                torch.nn.utils.clip_grad_norm_(
                    list(model.encoder.parameters()) + list(model.predictor.parameters()), max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # EMA update — use global_opt_step so momentum anneals over full training
                momentum = model.get_current_momentum(global_opt_step, TOTAL_OPT_STEPS)
                model.update_target_encoder(momentum)

                global_opt_step += 1

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            wandb.log({
                'loss': loss.item(),
                'prediction_loss': prediction_loss.item(),
                'sigreg': sigreg.item(),
                'cur_lr': optimizer.param_groups[0]['lr'],
                'cur_wd': optimizer.param_groups[0]['weight_decay'],
                'curr_mom': model.get_current_momentum(global_opt_step, TOTAL_OPT_STEPS)
            })
            print(f'\r[{epoch}/{config.epochs} | {data_idx}/{n_datapoints}] loss={loss.item()}', end='')
            # if data_idx % 100 == 0 and data_idx != 0:
            #     print('\nsend to save data...\n')
            #     yield {
            #         'type': 'final',
            #         'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
            #     }
            # else:
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
    timeout=7200,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={"/root/stl10": stl10_volume}
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
            print('-----')
            print('-----')
            print('FINAL')
            print('-----')
            print('-----')
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