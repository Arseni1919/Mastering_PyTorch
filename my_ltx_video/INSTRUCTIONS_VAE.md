# my_ltx_video — VAE

Simplest paper-related Video-VAE from *LTX-Video* (arXiv 2501.00103): causal 3D convs, PixelNorm, patchify moved inside the encoder — kept shallow, clean-latent decode (no holistic denoising-decoder trick yet), MSE+KL loss only. Trained on the KTH Action dataset (6 classes). DiT comes later.

## 0. Config

- `side_size` (**32**)
- `num_frames` (**8**)
- `channels` (**1**) — KTH is grayscale
- `vae_patch_size` (**2**) — spatial patch size for the patchify step
- `num_downsample_stages` (**2**) — patchify counts as stage 1; 1 more downsample stage after it
- `vae_blocks_per_stage` (**1**)
- `vae_base_channels` (**16**)
- `vae_latent_channels` (**8**)
- `vae_kl_weight` (**0.3**)
- `num_temporal_downsample_stages` (**1**) — see section 13; how many of the `num_downsample_stages - 1` stages also halve time, in addition to halving space
- `batch_size` (**4**)
- `learning_rate` (**1e-4**)
- `n_epochs` (**15**)

Latent shape: `(channels=8, T=num_frames / 2^num_temporal_downsample_stages, H=W=side_size / (vae_patch_size * 2^(num_downsample_stages-1)))` — with the values above, `(8, 4, 8, 8)`.

## 1. Data — `get_data.py`

```txt
KTHActionDataset — init(root, num_frames, side_size):
  scan the folder structure, build a list of (filepath, label) pairs — one entry per video, label from which class folder it's in

KTHActionDataset — __len__:
  number of samples

KTHActionDataset — __getitem__(idx):
  open the video with a video-decoding library
  pick a random contiguous window of num_frames frames
  crop to square, resize to side_size, convert to grayscale
  rearrange into (channels, frames, height, width) order, scale pixel values to [0,1]
  return clip, label
```
- `torchvision.io.read_video` was removed upstream — you'll need a different video-decoding library.
- Normalize to `[-1,1]` in `train.py`, not here.
- Label goes unused for now (VAE is unconditional).

**Sanity check:**
```python
clip, label = dataset[0]
# expected: clip.shape == (1, 8, 32, 32)
```

## 2. PixelNorm — `[function]`

```txt
pixel_norm(x):
  compute the root-mean-square of x across the channel dimension, separately for every spatiotemporal location
  divide x by that value (plus a tiny epsilon, for numerical stability when it's near zero)
```
No learnable parameters at all — this is why it's a function, not a Module.

**Sanity check:**
```python
x = torch.rand(2, 8, 4, 4, 4)
out = pixel_norm(x)
# expected: out.shape == (2, 8, 4, 4, 4); (out**2).mean(dim=1) ≈ 1 everywhere
```

## 3. CasualConv3D — `[Module]`

```txt
CasualConv3D — init(in_channels, out_channels, kernel_size, stride, padding):
  store kernel_size's TIME component separately — you'll need it in forward
  a standard 3D conv layer, built from the passed kernel_size/stride/padding directly (the padding you pass in should already have 0 in the time slot)

CasualConv3D — forward(x):
  slice out x's first frame WITHOUT losing the time dimension (indexing vs. slicing behave differently here — pick the one that keeps it)
  repeat that first frame (kernel_size_t - 1) times along the time axis
  concatenate the repeated frames onto the FRONT of x, along time
  run the stored 3D conv on the result
```
- Pad with the *repeated first frame*, not zeros — avoids a fabricated black-frame discontinuity.
- How many frames to prepend is driven by the *temporal* kernel size specifically — not by the height/width kernel size, even if it's tempting to grab whichever's convenient.

**Sanity check:**
```python
x = torch.rand(2, 3, 5, 16, 16)
out = causal_conv3d(x)              # kernel_size=(3,3,3), stride=1, padding=(0,1,1)
# expected: out.shape == (2, out_channels, 5, 16, 16)
```

## 4. Patchify3D / Unpatchify3D — `[Module]` / `[function]`

```txt
Patchify3D — init(channels, vae_base_channels, vae_patch_size):
  one CasualConv3D: kernel_size = stride = (1, vae_patch_size, vae_patch_size), padding = (0,0,0), channels -> vae_base_channels

Patchify3D — forward(x):
  run x through that conv, return the result
```

```txt
unpatchify3d(h, patch_size, channels):                     # [function] — h's channel count going in is channels * patch_size^2
  read off h's shape: batch, channels_in, time, height, width
  split channels_in back into three separate axes: (channels, patch_size, patch_size)
  move the two patch-offset axes so each sits directly next to its corresponding spatial axis — a permute, not a reshape
  merge each (patch-offset, spatial) pair back into one enlarged spatial axis
```
- Getting the channel-splitting order wrong scrambles the reassembled frame — same class of bug as any patchify/unpatchify pair you've built before.

**Sanity check:**
```python
x = torch.rand(2, 1, 8, 32, 32)
patched = patchify3d(x)                              # vae_patch_size=2
# expected: patched.shape == (2, vae_base_channels, 8, 16, 16)

h = torch.rand(2, 4, 8, 16, 16)
out = unpatchify3d(h, patch_size=2, channels=1)
# expected: out.shape == (2, 1, 8, 32, 32)
```

## 5. ResBlock — `[Module]`

```txt
ResBlock — init(in_channels, out_channels):
  two CasualConv3D layers, 3x3x3 kernels: in_channels -> out_channels, then out_channels -> out_channels
  if in_channels != out_channels: one more CasualConv3D, 1x1x1, for the skip projection — otherwise none

ResBlock — forward(h):
  skip = h
  h = conv1(nonlinearity(pixel_norm(h)))
  h = conv2(nonlinearity(pixel_norm(h)))
  if channels changed: skip = skip_conv(skip)
  return h + skip
```
Pre-norm order (normalize before convolving) — same convention as `my_ddpm`'s `ResNetBlock`.

**Sanity check:**
```python
h = torch.rand(2, 16, 8, 16, 16)
out = res_block(h)                  # ResBlock(16, 32)
# expected: out.shape == (2, 32, 8, 16, 16) — only channels change
```

## 6. Downsample / Upsample — `[Module]`

```txt
Downsample — init(in_channels):
  one CasualConv3D: kernel_size = stride = (1, 2, 2), padding = (0,0,0) — space only, time untouched

Downsample — forward(h):
  run h through that conv

Upsample — init(in_channels):
  a nearest-neighbor upsample layer, scale factor (1, 2, 2)
  one CasualConv3D "refine" layer applied after it: kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1) — a real spatial kernel, not 1x1x1

Upsample — forward(h):
  h = upsample_layer(h)
  return refine_conv(h)
```
The refine conv needs a real receptive field to actually do anything — a 1x1x1 kernel can't smooth neighboring-pixel artifacts.

**Sanity check:**
```python
h = torch.rand(2, 16, 8, 16, 16)
down = downsample(h)
# expected: down.shape == (2, 16, 8, 8, 8)
up = upsample(down)
# expected: up.shape == (2, 16, 8, 16, 16)
```

## 7. Encoder — `[Module]`

```txt
Encoder — init(channels, vae_base_channels, vae_patch_size, num_downsample_stages, vae_blocks_per_stage, vae_latent_channels):
  patchify = Patchify3D(...)
  stages = a container that actually registers submodules (a plain list silently won't)
  c = vae_base_channels
  loop (num_downsample_stages - 1) times:
    loop vae_blocks_per_stage times: append a ResBlock — the first block of the stage doubles channels (c -> 2c), the rest keep it (2c -> 2c)
    append a Downsample(2c)
    c = 2c
  bottleneck = vae_blocks_per_stage more ResBlocks, all c -> c, in the same kind of registering container
  conv_mu = CasualConv3D, 1x1x1, c -> vae_latent_channels
  conv_logvar = CasualConv3D, 1x1x1, c -> 1                    # uniform logvar — one shared channel, not vae_latent_channels

Encoder — forward(x):
  h = patchify(x)
  run h through each entry in stages, in order
  run h through each entry in bottleneck, in order
  return conv_mu(h), conv_logvar(h)
```
`mu`/`logvar` stay as spatial-temporal grids the whole way through — never flatten them into one vector per sample. A transformer needs a grid of positions to attend over later; collapsing that away now would defeat the purpose.

**Sanity check:**
```python
x = torch.rand(2, 1, 8, 32, 32)
mu, logvar = encoder(x)
# expected: mu.shape == (2, 8, 8, 8, 8), logvar.shape == (2, 1, 8, 8, 8)
```

## 8. reparameterize — `[function]`

```txt
reparameterize(mu, logvar):
  convert logvar into a standard deviation
  sample fresh Gaussian noise, same shape as mu
  return mu plus noise scaled by that standard deviation
```

**Sanity check:**
```python
mu = torch.rand(2, 8, 8, 8, 8)
logvar = torch.rand(2, 1, 8, 8, 8)
z = reparameterize(mu, logvar)
# expected: z.shape == (2, 8, 8, 8, 8) — matches mu's shape, not logvar's
```

## 9. Decoder — `[Module]`

```txt
Decoder — init(channels, vae_base_channels, vae_patch_size, num_downsample_stages, vae_blocks_per_stage, vae_latent_channels):
  c = the bottleneck's channel width — work this out from the encoder's own doubling schedule (vae_base_channels doubled num_downsample_stages-1 times)
  conv_in = CasualConv3D, 1x1x1, vae_latent_channels -> c
  bottleneck = vae_blocks_per_stage ResBlocks, all c -> c, in a registering container
  stages = a registering container, built by looping (num_downsample_stages - 1) times:
    append an Upsample(c)
    loop vae_blocks_per_stage times: append a ResBlock — the LAST block of the stage halves channels (c -> c/2), the rest keep it; update c once that halving happens
  conv_out = CasualConv3D, 3x3x3, c -> channels * vae_patch_size^2

Decoder — forward(z):
  h = conv_in(z)
  run h through each bottleneck entry, in order
  run h through each stages entry, in order
  h = conv_out(h)
  h = tanh(h)                         # bound output to [-1,1], matching how x is normalized in training
  return unpatchify3d(h, vae_patch_size, channels)
```
Mirror the encoder's channel schedule exactly, in reverse. No timestep conditioning anywhere in this version — the decoder only ever sees a clean latent.
- Without the final `tanh`, `x_hat` is unbounded while `x` is normalized to `[-1,1]` — the reconstruction loss can then exceed what's even mathematically possible for a properly-bounded output, and training becomes much noisier than it needs to be.

**Sanity check:**
```python
z = torch.rand(2, 8, 8, 8, 8)
x_hat = decoder(z)
# expected: x_hat.shape == (2, 1, 8, 32, 32)
```

## 10. kl_divergence — `[function]`

```txt
kl_divergence(mu, logvar):
  the standard closed-form KL divergence between a Gaussian(mu, exp(logvar)) and the standard normal prior
  sum that across every non-batch dimension (channels, time, height, width) for each sample
  average the resulting per-sample values across the batch
```
`logvar` broadcasts from its single channel against `mu`'s full channel count before you sum.

**Sanity check:**
```python
mu = torch.zeros(2, 8, 8, 8, 8)
logvar = torch.zeros(2, 1, 8, 8, 8)
kl = kl_divergence(mu, logvar)
# expected: kl ≈ 0 — this is exactly the prior, zero divergence from itself
```

## 11. Training — `train.py`

```txt
train_step(x):
  x normalized to [-1,1] and moved to device
  mu, logvar = encoder(x)
  z = reparameterize(mu, logvar)
  x_hat = decoder(z)
  reconstruction loss: compare x_hat against x — mind the sum-vs-mean reduction choice, same lesson from my_VAE
  regularization loss: kl_divergence(mu, logvar)
  combine the two with vae_kl_weight, backprop, optimizer step
```
Log the two loss terms separately, not just their combined sum.

## 12. Inference — `run_inference.py`

```txt
reconstruction check:
  encode a real clip, decode mu directly (not a fresh reparameterize sample — deterministic, for a clean comparison)
  display original vs. reconstructed, frame by frame

prior sampling:
  sample pure Gaussian noise matching the latent shape from section 0
  decode it directly
  display the resulting frames
```
Prior sampling won't look like real motion yet — nothing has learned what a *valid* latent trajectory looks like. That's the DiT's job, later.

## 13. Temporal downsampling — prerequisite for DiT

The DiT needs a manageable token count, so `T` can't stay equal to `num_frames` all the way through like it does above. Every downsample/upsample stage already halves space; now some of them need to also halve time. `Downsample`/`Upsample` (section 6) already accept a `shrink_time`/`expand_time` flag — it's just always been `False`. This section wires it up per-stage instead.

```txt
Encoder — stages loop, same shape as section 7, one addition:
  loop i_downsample_stage from 0 to (num_downsample_stages - 2):
    ... build vae_blocks_per_stage ResBlocks exactly as before ...
    shrink_time = i_downsample_stage < num_temporal_downsample_stages
    downsample_layer = Downsample(2c, shrink_time=shrink_time)
    append it, c = 2c
```

```txt
Decoder — stages loop, same shape as section 9, one addition:
  the decoder's stage loop runs in the OPPOSITE order from the encoder's — its first
  iteration mirrors the encoder's LAST downsample stage, not its first
  loop i_upsample_stage from 0 to (num_downsample_stages - 2):
    matching_encoder_stage = (num_downsample_stages - 2) - i_upsample_stage
    expand_time = matching_encoder_stage < num_temporal_downsample_stages
    upsample = Upsample(c, expand_time=expand_time)
    ... rest of the stage (ResBlocks, channel halving) exactly as before ...
```
Every `Downsample` that shrinks time must be matched by an `Upsample` that expands time at the *mirrored* stage — get the mirroring backwards and the decoder's output `T` won't match the encoder's input `T`.

**Sanity check:**
```python
x = torch.rand(2, 1, 8, 32, 32)   # num_frames=8
encoder = Encoder(..., num_downsample_stages=2, num_temporal_downsample_stages=1, ...)
mu, logvar = encoder(x)
# expected: mu.shape == (2, 8, 4, 8, 8) — T halved once, H/W halved twice (patchify + 1 stage)
decoder = Decoder(..., num_downsample_stages=2, num_temporal_downsample_stages=1, ...)
x_hat = decoder(mu)
# expected: x_hat.shape == (2, 1, 8, 32, 32) — back to the original T
```
