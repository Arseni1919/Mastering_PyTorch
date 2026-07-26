# my_ddpm — Rebuild Instructions

A from-scratch spec for `my_ddpm/`. Pseudocode only — fill in the real PyTorch yourself. Each step also lists the traps that actually bit during the original build.

## 0. Config

Fields you'll need across the files below: `num_classes`, `class_emb_size`, `time_emb_size`, `num_groups`, `base_channels`, `num_layers`, `has_attn`, `side_size`, `n_epochs`, `batch_size`, `learning_rate`, `num_train_timesteps`, `onnx_path`.

**Constraint:** `side_size` must be divisible by `2^num_layers` (each down-stage halves spatial size; the up-path always doubles via a fixed-factor upsample, so an odd intermediate size makes the two paths disagree). MNIST is natively 28×28 — pad it to a clean multiple (e.g. 32) rather than fighting this.

## 1. Data (`get_data.py`)

```txt
load MNIST via torchvision.datasets.MNIST
transform: pad to side_size -> convert to tensor
wrap in a DataLoader
```

## 2. Model (`define_model.py`) — main focus

### 2.1 Sinusoidal timestep embedding
```txt
sinusoidal_embedding(t, dim):
  half = dim // 2
  freqs = geometrically-decaying frequencies, length `half`   # like transformer positional encoding
  args = t (reshaped to broadcast) * freqs
  return concat(sin(args), cos(args))
```
**Gotchas:** `t` arrives as shape `(bs,)` — you must add a trailing dim (`(bs,1)`) before multiplying against `freqs` (`(half,)`), or you'll silently broadcast wrong. `freqs` is built fresh each call — make sure it ends up on the same device as `t`, or you'll get a device-mismatch error the moment you leave CPU-only testing.

### 2.2 Time embedding MLP
```txt
small MLP: time_emb_size -> 4x hidden -> time_emb_size, nonlinearity in between
```

### 2.3 ResNetBlock (the workhorse — own nn.Module, instantiated many times with different channel counts)
```txt
resnet_block(x, t_emb):
  h = normalize(x) -> nonlinearity -> spatial_conv(h)      # in_channels -> out_channels, pad to keep H,W
  h = h + project(t_emb) broadcast over spatial dims        # inject time
  h = normalize(h) -> nonlinearity -> spatial_conv(h)       # out_channels -> out_channels, pad to keep H,W
  return h + (x, or a 1x1-conv projection of x if channels changed)
```
**Gotchas:** two separate norm layers (different channel counts, so different instances). `padding=1` with `kernel_size=3` is what keeps H/W unchanged — solve from the conv output-size formula if unsure. Time projection needs reshaping to `(bs, out_channels, 1, 1)` before the add. The residual add needs a 1×1 conv on the skip path *only* when `in_channels != out_channels`; a 1×1 conv changes channels without mixing neighboring pixels.

### 2.4 AttnBlock (own nn.Module)
```txt
attn_block(x):
  h = normalize(x)
  q, k, v = per-pixel projections of h (1x1 convs, no spatial mixing)
  reshape q,k,v so spatial positions become a sequence: (bs, channels, H, W) -> (bs, H*W, channels)
  attn_weights = softmax(q @ k^T / sqrt(channels))
  h = attn_weights @ v, reshape back to (bs, channels, H, W)
  return x + project_out(h)
```
**Gotchas:** the reshape is NOT a plain `.view()` — channels-first to channels-last requires actually moving the axis (`permute`) before flattening, or the values end up scrambled. `.view()`/`.reshape()` alone just reslices the flat memory buffer, it doesn't reorder data.

### 2.5 DownBlock (own nn.Module, wraps one ResNetBlock + optional AttnBlock)
```txt
down_block(x, t_emb, has_attn):
  x = resnet_block(x, t_emb)
  if has_attn: x = attn_block(x)
  skip = x                       # save BEFORE downsampling
  x = downsample(x)              # stride-2 conv, halves H,W, channel count unchanged
  return x, skip
```

### 2.6 MidBlock (own nn.Module, always has attention)
```txt
mid_block(x, t_emb):
  x = resnet_block(x, t_emb)
  x = attn_block(x)
  x = resnet_block(x, t_emb)
  return x
```

### 2.7 UpBlock (own nn.Module, mirrors DownBlock)
```txt
up_block(x, skip, t_emb, has_attn):
  x = upsample(x)                     # doubles H,W (e.g. nearest-neighbor)
  x = refine_conv(x)                  # 3x3 conv, same channel count, keeps H,W — smooths upsample artifacts
  x = concat([x, skip], dim=channel)  # x and skip now have the same spatial size
  x = resnet_block(x, t_emb)          # in_channels = concatenated width, out_channels = target width
  if has_attn: x = attn_block(x)
  return x
```
**Gotchas:** upsample-then-concat only works if `skip`'s spatial size exactly matches the upsampled `x` — this is exactly the `side_size`/`num_layers` constraint from section 0. The `resnet_block`'s input channel count is the *sum* of `x`'s and `skip`'s channels after concat, not just one of them.

### 2.8 Full model wiring
```txt
unet_forward(x, t, class_labels):
  class_cond = embed(class_labels), broadcast to spatial dims, concat onto x (channel dim)
  t_emb = time_embed(sinusoidal_embedding(t, time_emb_size))

  x = conv_in(x)                          # (image_channels + class_emb_size) -> base_channels

  skips = []
  for i in range(num_layers):             # channels double each stage: base*2^i -> base*2^(i+1)
      x, s = down_block(x, t_emb)
      skips.append(s)

  x = mid_block(x, t_emb)                 # at base * 2^num_layers channels

  for i in reversed(range(num_layers)):   # mirrored, channels halve back down
      x = up_block(x, skips.pop(), t_emb)

  return conv_out(x)                      # base_channels -> image_channels
```

## 3. Training (`train.py`)

```txt
device = pick cuda, else mps, else cpu
dataloader = DataLoader over the dataset
noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps, beta_schedule='squaredcos_cap_v2')
net, optimizer (Adam), loss_fn (MSELoss)

for epoch in range(n_epochs):
  for x, y in dataloader:
    x = x * 2 - 1                                   # normalize to [-1, 1]
    noise = standard Gaussian noise, same shape as x  # NOT uniform
    t = random integer timesteps, one per sample, shape (bs,)
    noisy_x = noise_scheduler.add_noise(x, noise, t)
    pred = net(noisy_x, t, y)
    loss = MSE(pred, noise)                          # predict the noise, not the clean image
    backward + optimizer step
  log epoch average loss
```
**Gotchas:** noise must be Gaussian (`randn`, not `rand`) — the whole diffusion math assumes it. If wrapping training in a generator for remote/streaming execution (e.g. Modal), never yield a live GPU tensor or call `.cpu()` on the actual model mid-training (it mutates the model in place) — copy each `state_dict` tensor to CPU individually instead (`{k: v.cpu() for k, v in net.state_dict().items()}`).

## 4. Inference (`run_inference.py`)

```txt
load net, load_state_dict from the saved checkpoint (map_location=device), net.eval()
noise_scheduler = same DDPMScheduler config as training

x = pure Gaussian noise, shape (num_samples, channels, H, W)
y = desired class labels

for t in noise_scheduler.timesteps:        # descends from ~num_train_timesteps to 0
  t_for_model = t expanded to shape (bs,), moved to device
  pred = net(x, t_for_model, y)             # no_grad
  x = noise_scheduler.step(pred, t, x).prev_sample   # note: t here, NOT t_for_model

rescale x from [-1,1] back to [0,1] for display
```
**Gotchas:** `net(...)` and `noise_scheduler.step(...)` need `t` in *different* shapes — the model needs `(bs,)` on-device, but the scheduler's internal `alphas_cumprod` lookup wants the original scalar `t` as yielded by `noise_scheduler.timesteps`, unmodified. Keep both variables around, don't overwrite one with the other.

## 5. Deploy — ONNX export only (`deploy.py`)

```txt
# export once, offline (not in the hot loop):
torch.onnx.export(net, (example_x, example_t, example_y), f=onnx_path)

# then for inference:
session = onnxruntime.InferenceSession(onnx_path)
input_names = session's input names
output_names = session's output names

per step:
  feed = {input_names[i]: corresponding tensor.cpu().numpy() for each of x, t, y}
  outputs = session.run(output_names, feed)     # returns a list of numpy arrays
  pred = torch.from_numpy(outputs[0]).to(device)
```
**Gotcha:** PyTorch's exporter may write model weights to a *separate* companion file (`<name>.onnx.data`) next to the `.onnx` graph file rather than embedding them inline. If deploying remotely, both files must be shipped — the graph file alone will fail to load.
