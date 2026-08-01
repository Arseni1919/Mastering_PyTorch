# my_VAE — Rebuild Instructions

A from-scratch spec for `my_VAE/`: an unconditional Variational Autoencoder trained on CIFAR-10. Pure PyTorch. Pseudocode only — fill in the real PyTorch yourself.

## 0. Config

Fields you'll need: `channels` (3 for CIFAR-10), `side_size` (32), `base_channels`, `latent_dim`, `n_epochs`, `batch_size`, `learning_rate`, `kl_weight` (weighting on the KL term, aka β in β-VAE), `num_interpolation_steps` (for the inference interpolation test).

**Constraint:** however many stride-2 downsampling stages the encoder uses, `side_size` must be evenly divisible by `2^num_stages` — same reasoning as the UNet/DiT constraints in the other projects, just applied to the encoder/decoder's conv stack here instead.

## 1. Data (`get_data.py`)

```txt
load CIFAR-10 via torchvision.datasets.CIFAR10
transform: convert to tensor   # already 32x32, no padding needed
wrap in a DataLoader
```
Unlike `my_ddpm`/`my_dit`, this is unconditional — the label CIFAR-10 provides per image is never used anywhere in this project.

## 2. Model (`define_model.py`)

**Module vs. function rule** (same as `my_dit`): learnable-parameter-owning pieces are `nn.Module`s; pure computation is a plain function.

### 2.1 Encoder — `[Module]`
```txt
encoder(x):                              # x: (bs, channels, side, side)
  h = conv stack, stride-2 each stage, doubling channels, with a nonlinearity between stages
  h = flatten(h)                         # (bs, flat_dim)
  mu = linear(h)                         # (bs, latent_dim) — no nonlinearity on this output
  logvar = linear(h)                     # (bs, latent_dim) — no nonlinearity on this output
  return mu, logvar
```
**Gotcha:** `flat_dim` depends on how many stride-2 stages you use — work it out as `base_channels_at_final_stage * (side_size / 2^num_stages)^2` before wiring up the `Linear` layers, the same shape-tracing discipline as the UNet/DiT builds. Don't apply an activation to `mu`/`logvar` themselves — they need to span the full real line (this matters especially for `logvar`, which can legitimately be negative).

### 2.2 Reparameterization — `[function]` (pure computation, no learnable weights)
```txt
reparameterize(mu, logvar):
  std = exp(0.5 * logvar)
  eps = standard Gaussian noise, same shape as std
  return mu + eps * std
```
**Gotcha:** this is the entire trick that makes VAEs trainable — sampling `eps` happens *outside* the computation graph (it doesn't depend on any learnable parameter), while `mu + eps * std` is a plain differentiable affine operation. If you instead sampled `z` directly from `N(mu, std)` in one step, gradients couldn't flow back through the randomness into the encoder.

### 2.3 Decoder — `[Module]`, mirrors the encoder in reverse
```txt
decoder(z):                              # z: (bs, latent_dim)
  h = linear(z)                          # -> flat_dim
  h = reshape to the encoder's final conv shape (channels, side/2^num_stages, side/2^num_stages)
  h = conv stack, upsampling by 2 each stage, halving channels, mirroring the encoder's schedule in reverse
  return final_conv(h)                   # -> channels, with Tanh (matches [-1,1]-normalized input)
```
**Gotcha:** same upsample-artifact tradeoff as `my_ddpm`'s `UpBlock` — prefer `Upsample` + `Conv2d` over `ConvTranspose2d` to avoid checkerboard patterns. The final activation must match however you normalize the input in `train.py` — `Tanh` if `x` is scaled to `[-1, 1]`, `Sigmoid` if left in `[0, 1]`.

### 2.4 KL divergence — `[function]` (pure computation, no learnable weights)
```txt
kl_divergence(mu, logvar):
  per_sample_kl = -0.5 * sum(1 + logvar - mu^2 - exp(logvar), over the latent dimension)
  return mean(per_sample_kl, over the batch)
```
**Gotcha:** sum over the latent dimension *first*, then average over the batch — don't average over every element (batch × latent_dim) flattened together. Get this wrong and the effective balance between reconstruction loss and KL (relative to `kl_weight`) silently shifts with `latent_dim`, making your `kl_weight` non-portable across latent sizes.

### 2.5 Full VAE wiring — `[Module]`
```txt
vae_forward(x):
  mu, logvar = encoder(x)
  z = reparameterize(mu, logvar)
  x_hat = decoder(z)
  return x_hat, mu, logvar
```

## 3. Training (`train.py`)

```txt
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
```
**Gotchas:** log `recon_loss` and `kl_loss` as two separate numbers, not just their sum — this is how you catch **posterior collapse** (`kl_loss` collapsing toward 0, meaning the decoder learned to ignore `z` and just reconstruct from nothing) versus a reconstruction that simply isn't learning. Also carried over from `my_ddpm`/`my_dit`: don't forget `.to(device)` on `x`, and zero gradients before `backward()`.

## 4. Inference (`run_inference.py`) — three distinct checks, not one

```txt
# 1. Prior sampling — pure generation, single decoder pass, no iterative loop needed (unlike diffusion/flow matching)
z = standard Gaussian noise, shape (num_samples, latent_dim)
images = decoder(z)

# 2. Reconstruction check — encode real images, decode back, compare to originals
mu, logvar = encoder(real_images)
reconstructed = decoder(mu)              # use mu directly, not reparameterize() — see gotcha

# 3. Latent interpolation — the classic "prove the latent space is smooth" demo
mu_a, _ = encoder(image_a)
mu_b, _ = encoder(image_b)
for step in range(num_interpolation_steps):
  alpha = step / (num_interpolation_steps - 1)
  z_interp = (1 - alpha) * mu_a + alpha * mu_b
  images.append(decoder(z_interp))
```
**Gotcha:** for reconstruction and interpolation, decode `mu` directly rather than calling `reparameterize()` — you want the deterministic "best guess" latent for a given input at evaluation time, not a noisy sample. Sampling `eps` is only needed during *training* (so gradients can reach the encoder) and for *prior sampling* (where there's no input image to encode in the first place, so you sample fresh from `N(0,I)` directly).

## 5. Latent health checks

- Watch `kl_loss` over training — collapsing toward 0 means posterior collapse (decoder ignoring the latent); it shouldn't vanish, and it shouldn't dominate the loss so much that reconstructions go blank either.
- Across a validation batch, check the empirical mean/std of `mu` (and of a full reparameterized `z`) against the prior's `N(0,1)` — this matters more here than it would in a study of the VAE alone, because phase 3's DiT will be trained to model whatever distribution this latent space actually has.
