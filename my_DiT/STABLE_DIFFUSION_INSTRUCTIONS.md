# my_stable_diffusion — Rebuild Instructions

Phase 3: a latent flow-matching pipeline on CelebA (128×128), using a **frozen, pretrained** VAE (`diffusers`) for compression and a DiT (mostly reused from `my_DiT`) operating entirely on its latent grid. Pseudocode only for the parts that are genuinely new — for the DiT internals that carry over unchanged, this points back at what you already built rather than re-deriving them.

**Design decision, flag if you disagree:** CelebA has no clean single-class label the way MNIST's digits did (it has 40 multi-label attributes, a different conditioning problem entirely). This pipeline is **unconditional** — only timestep conditioning, no class embedding. If you want attribute-conditioning later, that's a deliberate follow-up, not part of this pass.

## 0. Config

Fields you'll need: `side_size` (128, pixel-space image size), `vae_scale_factor` (8 — fixed by the pretrained VAE's architecture, not something you choose), `latent_channels` (4 — also fixed by the pretrained VAE), `pretrained_vae_id` (the `diffusers` model string, e.g. `"stabilityai/sd-vae-ft-mse"`), `patch_size` (for the DiT, operating on the *latent* grid), `hidden_size`, `num_heads`, `num_layers`, `mlp_ratio`, `time_emb_size`, `n_epochs`, `batch_size`, `learning_rate`, `num_inference_steps`.

**Constraint:** `side_size` must be divisible by `vae_scale_factor` (128/8 = 16 ✓). The resulting `latent_side_size = side_size // vae_scale_factor` must itself be divisible by `patch_size` — same divisibility discipline as `my_DiT`, just one level removed: it's the *latent* grid that needs to tile evenly, not the original image.

## 1. Data (`../my_stable_diffusion/get_data.py`)

```txt
load CelebA via torchvision.datasets.CelebA
transform: Resize(side_size) -> CenterCrop(side_size) -> ToTensor()
wrap in a DataLoader
```
Like `my_VAE`, the labels CelebA provides (identity, 40 binary attributes) are never used — this is unconditional. **Known gotcha, not yours to fix:** `torchvision`'s `CelebA` downloader pulls from Google Drive and frequently hits quota limits / fails outright. If it fails, that's the well-known flakiness, not a bug in your code — you may need to retry, use a mirror, or download manually.

## 2. Pretrained VAE wrapper (`../my_stable_diffusion/vae.py`)

This is real, direct `diffusers` API — not pseudocode, since getting these details wrong is the most common way people break latent diffusion on their first attempt:

```python
from diffusers import AutoencoderKL

def load_frozen_vae(pretrained_vae_id, device):
    vae = AutoencoderKL.from_pretrained(pretrained_vae_id).to(device)
    vae.requires_grad_(False)
    vae.eval()
    return vae

def encode_to_latents(vae, x):
    # x: (bs, 3, side, side), expected in [-1, 1] — matches your existing normalization convention
    posterior = vae.encode(x).latent_dist
    latents = posterior.sample()                    # reparameterized sample — same trick you implemented by hand in my_VAE
    return latents * vae.config.scaling_factor       # see gotcha below

def decode_from_latents(vae, z):
    z = z / vae.config.scaling_factor                # undo the scaling before decoding
    return vae.decode(z).sample                       # note: .decode(...).sample, not the return value directly
```
**Gotchas:**
- `vae.config.scaling_factor` (typically `0.18215` for SD1.x VAEs) rescales the raw latent to roughly unit variance before it's used for diffusion/flow-matching training — this is a real, easy-to-miss step. Skip it and your `x_1` (real latent) has a different scale than `x_0` (unit-variance Gaussian noise), which quietly hurts training. Multiply by it when encoding, divide by it when decoding — inverse operations, applied at opposite ends of the pipeline.
- `vae.encode(x)` returns an object with `.latent_dist` (a distribution, not a tensor directly) — `.sample()` reparameterizes like your own `reparameterize()`; `.mean` gives the deterministic version if you ever want that instead (e.g. for inspecting reconstructions without sampling noise).
- `vae.decode(z)` returns an object too — the actual image tensor is `.sample`, not the return value itself.
- Wrap encode calls in `torch.no_grad()` in `../my_stable_diffusion/train.py` — the VAE is frozen, so building a gradient graph through it is pure waste.
- This VAE is a completely separate object from your DiT — it doesn't belong inside the trainable model's `state_dict`. Load it once in `../my_stable_diffusion/train.py`/`../my_stable_diffusion/run_inference.py`, pass it (or the already-encoded latents) around explicitly, and only save/load the DiT's own weights as checkpoints.

## 3. Model (`../my_stable_diffusion/define_model.py`)

**Almost everything here is a straight copy from `define_model.py` — the DiT you already built doesn't know or care whether its input is raw pixels or VAE latents, since `Patchify`'s `nn.Conv2d` and `unpatchify`'s reshape logic are generic over channel count and spatial size.** Copy these over unchanged: `build_2d_rope`, `modulate`, `apply_rope`, `Patchify`, `TimeEmbedding`, `AdaLN`, `MultiHeadAttnBlock`, `DiTBlock`, `FinalLayer`, `unpatchify`.

**What actually changes — the top-level wiring class** (call it `LatentDiTModel`, adapted from `DiTFullModel`):
```txt
# in __init__:
#   channels = latent_channels (4), not the original image's 3
#   side = latent_side_size (side_size // vae_scale_factor), not side_size
#   rope_cos, rope_sin = build_2d_rope(side // patch_size, head_dim)   # unchanged mechanism, smaller grid
#   NO class_embedding at all — this pipeline is unconditional

dit_forward(x, t):                          # note: no class_labels argument anymore
  tokens = patchify(x)
  cond = time_embed(t)                      # was time_embed(t) + class_embed(class_labels) in my_DiT — class term is just gone
  for block in blocks:
    tokens = block(tokens, cond, rope_cos, rope_sin)
  tokens = final_layer(tokens, cond)
  return unpatchify(tokens)
```
**Gotcha:** `cond_dim` (fed into `AdaLN`/`FinalLayer`) is now just `time_emb_size` — there's no `class_emb_size` term to add anymore. Don't leave a stale `+ class_emb_size` in the sizing math, or your `AdaLN`/`FinalLayer` `Linear` layers will be sized for a `cond` vector that's larger than what `forward` actually produces.

## 4. Training (`../my_stable_diffusion/train.py`) — flow matching, in latent space

```txt
device = pick cuda, else mps, else cpu
dataloader = DataLoader over the dataset
vae = load_frozen_vae(pretrained_vae_id, device)
net = LatentDiTModel(...).to(device)
optimizer (Adam, lr=learning_rate), loss_fn (MSELoss)

for epoch in range(n_epochs):
  for x, _ in dataloader:                    # label unused — unconditional
    x = x.to(device) * 2 - 1                 # normalize to [-1, 1] BEFORE encoding — the VAE expects this range
    with torch.no_grad():
      x_1 = encode_to_latents(vae, x)        # real image -> real latent; this replaces raw pixels as your training target
    bs = x_1.shape[0]
    t = Uniform(0, 1), one independent value per sample, shape (bs,)
    reshaped_t = t reshaped to (bs, 1, 1, 1)
    x_0 = standard Gaussian noise, same shape as x_1               # NOT same shape as x — shape of the LATENT
    x_t = (1 - reshaped_t) * x_0 + reshaped_t * x_1
    target = x_1 - x_0
    pred = net(x_t, t)                        # no class_labels argument
    loss = MSE(pred, target)
    zero_grad + backward + optimizer step
  log epoch average loss
```
**Gotchas, all carried forward from `my_DiT` plus one new one:** `t` independent per sample, not a shared scalar. Two shapes of `t` needed simultaneously (`(bs,1,1,1)` vs `(bs,)`) — same as before. `net` and data both need `.to(device)` — same as before. **New:** `x_0`'s shape must match the *latent's* shape (`(bs, latent_channels, latent_side_size, latent_side_size)`), not the original image's shape — an easy copy-paste mistake if you're adapting straight from `my_DiT`'s training loop, since there `x_0` was shaped like the raw image.

## 5. Inference (`../my_stable_diffusion/run_inference.py`) — Euler integration in latent space, then decode once

```txt
vae = load_frozen_vae(pretrained_vae_id, device)
net = LatentDiTModel(...), load checkpoint, eval()

x = pure Gaussian noise, shape (num_samples, latent_channels, latent_side_size, latent_side_size)
t = 0.0
step_size = 1 / num_inference_steps

for _ in range(num_inference_steps):
  t_tensor = a (bs,)-shaped tensor filled with the current t, on device
  pred = net(x, t_tensor)                    # no_grad, no class_labels
  x = x + step_size * pred
  t += step_size

with torch.no_grad():
  images = decode_from_latents(vae, x)       # the ONLY point pixels re-appear — single decoder call, not per-step

rescale images from [-1,1] back to [0,1] for display
```
**Gotcha:** the entire iterative loop stays in latent space — the VAE decoder is called exactly once, after the loop finishes, not at every step. Calling it every step would be both wasteful and conceptually wrong (the intermediate `x` values during the loop are noisy latents, not something the decoder is meant to interpret as a real image).
