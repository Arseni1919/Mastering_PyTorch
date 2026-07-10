# 02-ddpm: split into dataset/train/inference + W&B

## Context

`02-ddpm/` already has a working DDPM implementation: `model.py` defines a class-conditional
UNet with attention and a `Diffusion` module (q_sample, p_losses, DDPM/DDIM samplers, CFG);
`train.py` is a Modal entrypoint that trains on CIFAR-10 on an H100 and writes checkpoints and
sample grids to a Modal volume; `utils.py` holds the shared Modal harness (`app`, `image`,
`volume`, `load_impl`, `seed_everything`, `save_grid`).

What's missing: the CIFAR-10 loader is inline in `train.py` instead of its own file, there's no
standalone inference script, checkpoints only live on the Modal volume (never make it to the
local machine), and there's no experiment tracking.

## Goals

- Split responsibilities into: dataset/dataloader, train loop, model definition (unchanged),
  inference.
- After a training run, the checkpoint and artifacts end up in the local `02-ddpm/` folder, not
  just on the Modal volume.
- A separate script loads the saved checkpoint and generates images, without needing Modal.
- Training logs to Weights & Biases (project `ddpm-cifar10`): loss curve + periodic sample grids.

## Non-goals

- Changing `model.py`'s architecture or public surface. It stays byte-for-byte compatible with
  `tests/test_ddpm.py`'s `IMPL`-oracle pattern (`UNet`, `Diffusion`, `cosine_abar`,
  `timestep_embedding`, etc. all still live in one file).
- Adding a `template.py` oracle for the new files. The existing test suite only exercises
  `model.py`; `dataset.py`, `train.py`, and `inference.py` stay untested, consistent with how
  `train.py` is untested today (Modal/W&B/filesystem-heavy, not unit-test material).
- Local (non-Modal) training. Training stays on Modal's H100 to hit the "real result within an
  hour" goal from the top-level design; local training would be 20-50x slower on a Mac.

## Architecture

```
02-ddpm/
  model.py       unchanged: UNet, Diffusion, building blocks, cosine_abar, timestep_embedding
  dataset.py     new: cifar_loader(batch_size) -> DataLoader
  train.py       refactored: Modal entrypoint, train loop, W&B logging, local artifact download
  inference.py   new: local script, loads a checkpoint and generates images
  utils.py       extended: existing Modal harness + pick_device() for local inference
  tests/         unchanged
```

## Components

### `dataset.py`

Moves `cifar_loader(batch_size)` out of `train.py` verbatim (CIFAR-10 via torchvision, random
horizontal flip, normalize to [-1, 1], download to `/vol/data`). Only used inside the Modal
container, so it keeps referencing `/vol/data`.

### `train.py`

Keeps its current shape (Modal `@app.function` + `@app.local_entrypoint`) but:

- Imports `cifar_loader` from `dataset.py` instead of defining it inline.
- Calls `wandb.init(project="ddpm-cifar10", name="02-ddpm", config={...})` at the start of the
  remote `train()` function, using a Modal secret named `wandb` (holding `WANDB_API_KEY`)
  attached via `app.function(..., secrets=[modal.Secret.from_name("wandb")])`.
- Logs `{"loss": avg}` every 200 steps (same cadence as the existing print) via `wandb.log`.
- Logs a sample grid as `wandb.Image` periodically during training (every ~5 minutes, reusing
  the EMA model for sampling) so generation quality is visible mid-run.
- At the end, logs the final CFG sweep grid and loss plot to W&B same as it writes them to the
  volume today, then calls `wandb.finish()`.
- In the `local_entrypoint`, after `train.remote(...)` returns, reads `ema.pt`, `samples.png`,
  `cfg_sweep.png`, and `loss.png` back from the Modal volume (`volume.read_file(...)`) and writes
  each into the local `02-ddpm/` folder, so the run's outputs land next to the code without a
  manual `modal volume get`.

### `inference.py`

Plain local script, no Modal:

- CLI args: `--labels` (comma-separated CIFAR-10 class indices), `--n` (samples per label),
  `--steps` (DDIM steps, default 50), `--cfg` (guidance weight, default 3.0), `--ckpt` (default
  `02-ddpm/ema.pt`), `--out` (default `02-ddpm/generated.png`).
- Loads the checkpoint with `torch.load`, builds `UNet` + `Diffusion`, loads the state dict,
  moves to `pick_device()`.
- Runs `Diffusion.ddim_sample` for the requested labels and saves a grid via `save_grid` (moved
  usage, function itself already lives in `utils.py`).

### `utils.py`

Adds `pick_device()` (`cuda` if available, else `mps`, else `cpu`) for `inference.py`. Everything
else (`app`, `image`, `volume`, `load_impl`, `seed_everything`, `save_grid`) is unchanged. `image`
gains `wandb` in its `uv_pip_install` list.

## Data flow

**Training**: `uv run modal run 02-ddpm/train.py` → local entrypoint calls `train.remote()` → on
the H100: `dataset.cifar_loader` feeds the loop, `wandb.init`/`wandb.log` track loss and sample
grids, EMA checkpoint + artifacts are written to the Modal volume as today → local entrypoint
downloads those same files into `02-ddpm/` before exiting.

**Inference**: `uv run python 02-ddpm/inference.py --labels 0,3,7 --n 8` → loads
`02-ddpm/ema.pt` → samples on the local device → writes `02-ddpm/generated.png`.

## Testing

No new automated tests. `tests/test_ddpm.py` is untouched since `model.py`'s public surface
doesn't change. `dataset.py`, the refactored `train.py`, and `inference.py` are exercised
manually (a real Modal run, then a real local inference run) rather than unit-tested, matching
how `train.py` is handled today.

## Setup required outside the code

- `modal secret create wandb WANDB_API_KEY=<key>` — one-time, before the first training run.
