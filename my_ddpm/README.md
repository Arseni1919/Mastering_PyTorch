# DDPM

## Architecture

![unet2d_architecture.png](pics/unet2d_architecture.png)

The core idea: it's a symmetric encoder-decoder where every down-block saves its output before downsampling, and the corresponding up-block concatenates that saved tensor before its own convs. Timestep info gets injected into every ResNet block along the way (not drawn — it'd clutter the diagram, but it matters).

Here's the pseudo-code skeleton, concise enough to leave the real implementation work to you:

```txt
# --- Timestep embedding ---
def time_embed(t):
    e = sinusoidal_embedding(t, dim)      # like transformer positional encoding
    e = Linear(e) -> SiLU -> Linear(e)    # small MLP, dim -> 4*dim -> dim
    return e

# --- ResNet block (the workhorse, repeated everywhere) ---
def resnet_block(x, t_emb):
    h = GroupNorm(x) -> SiLU -> Conv3x3(h)
    h = h + Linear(t_emb)[:, :, None, None]   # inject time info
    h = GroupNorm(h) -> SiLU -> Conv3x3(h)
    return h + skip_conv(x)   # skip_conv = identity or 1x1 conv if channels change

# --- Attention block (only at lower resolutions) ---
def attn_block(x):
    h = GroupNorm(x)
    q, k, v = project(h)               # 1x1 convs
    h = softmax(q @ k.T / sqrt(d)) @ v  # standard self-attention over spatial positions
    return x + project_out(h)

# --- Down block ---
def down_block(x, t_emb, has_attn):
    x = resnet_block(x, t_emb)
    if has_attn: x = attn_block(x)
    skip = x                      # SAVE this for the matching up block
    x = downsample(x)             # strided conv or avgpool, halves H,W
    return x, skip

# --- Mid block ---
def mid_block(x, t_emb):
    x = resnet_block(x, t_emb)
    x = attn_block(x)
    x = resnet_block(x, t_emb)
    return x

# --- Up block ---
def up_block(x, skip, t_emb, has_attn):
    x = upsample(x)                    # nearest/transpose conv, doubles H,W
    x = concat([x, skip], dim=channel) # this is the U-Net magic
    x = resnet_block(x, t_emb)
    if has_attn: x = attn_block(x)
    return x

# --- Full forward pass ---
def unet_forward(x, t):
    t_emb = time_embed(t)
    x = conv_in(x)

    skips = []
    for block in down_blocks:
        x, s = down_block(x, t_emb, ...)
        skips.append(s)

    x = mid_block(x, t_emb)

    for block in up_blocks:
        x = up_block(x, skips.pop(), t_emb, ...)   # pop in reverse order

    return conv_out(x)   # predicts noise, same shape as input
```

A few things that trip people up when reimplementing:

- Channel counts double each down-block and halve each up-block (e.g. 64→128→256→256, mirrored back).
- Skip concatenation means the up-block's resnet input channels = upsampled_channels + skip_channels, not just upsampled_channels — easy to get a shape mismatch here.
- Attention is usually only applied at the lower spatial resolutions (e.g. 16×16, 8×8), not every block — it's O(N²) in pixel count.
- The time embedding MLP output gets added (not concatenated) inside every single ResNet block, both down and up.

## TODO

- [x] **Time embedding MLP** — small network that projects the sinusoidal time embedding through a wider hidden layer and back down, with a nonlinearity in between. Fix `t`'s shape wherever it's currently wrong.
- [x] **ResNet block** — norm/activation/conv, inject time embedding, norm/activation/conv again, residual add (with a projection on the skip path if channel count changes).
- [x] **Attention block** — self-attention over spatial positions, used only at lower resolutions.
- [x] **Down block** — resnet block (+ optional attention) → save the output as a skip connection → downsample.
- [x] **Mid block** — resnet → attention → resnet, at the bottleneck resolution.
- [x] **Up block** — upsample → concatenate the matching skip connection → resnet block (+ optional attention).
- [x] **Wire the full UNet** — `conv_in`, the stack of down blocks, mid block, stack of up blocks, `conv_out`; decide the channel schedule and where attention kicks in.
- [x] **Shape-check the full forward pass** end to end.

## Training TODO

- [x] **Setup** — device selection, training hyperparameters in `config.py`.
- [x] **Data** — `DataLoader` over the existing MNIST `dataset`.
- [x] **Noise scheduler** — instantiate `diffusers.DDPMScheduler`.
- [x] **Model / optimizer / loss** — instantiate the model, `Adam` optimizer, `MSELoss`.
- [x] **Single training step** — sample noise + timestep, corrupt the image, predict, compute loss, backprop, optimizer step.
- [x] **Epoch loop + wandb logging** — wrap the training step over the dataloader and epochs, log metrics.
- [x] **Modal wrapping** — `modal.Image`, `modal.App`, GPU-decorated remote training function, local entrypoint.
- [x] **Checkpointing + loss visualization** — save `state_dict`, plot losses.

## Inference TODO

- [x] **Load model + scheduler** — instantiate the model, load trained weights from `saved_model.pt`, instantiate the matching `DDPMScheduler`.
- [x] **Initialize sampling** — start from pure noise, prepare desired class labels.
- [x] **Reverse denoising loop** — step backward through every timestep, predicting noise and removing it via the scheduler.
- [x] **Visualization** — display/save the generated digit images.
- [x] **Modal wrapping** — GPU-remote pattern reused from `train.py`.

## Deploy TODO (implemented directly in `run_inference.py`, not a separate file)

- [x] **Baseline benchmark** — measure current model forward-pass latency before optimizing. Result: mean ≈ 4.76ms, std ≈ 0.59ms, max ≈ 9.77ms per step (post-warmup). Memory measurement skipped by choice.
- [x] **Profiling** — find the actual bottleneck layers/blocks inside the network. Top findings: `aten::upsample_nearest2d` is the single biggest self-time consumer (29%), device-transfer ops (`aten::to`/`copy_`) surprisingly high (~22%), convs ~21-25%.
- [x] **`torch.compile`** — tried, made things worse (mean 9.18ms vs 4.86ms baseline) on MPS with this small model; kept disabled. Likely causes: MPS backend's less mature kernel fusion, guard-check overhead not offset by fusion gains, model too small to have much dispatch overhead to amortize away.
- [x] **Reduced precision** — tried `torch.autocast(dtype=torch.float16)`, made things worse both on MPS (5.53ms vs 4.86ms baseline) and on real CUDA (A10G via Modal); kept disabled. Same root cause as `torch.compile`: model too small for the per-op overhead to pay for itself, regardless of backend — not an MPS-specific limitation.
- [x] **Quantization** — used torchao's `Int8DynamicActivationInt8WeightConfig` (up-to-date replacement for deprecated `torch.quantization.quantize_dynamic`). Tried on real Modal CUDA GPU: dramatically worse (mean 37.4ms vs 3.34ms baseline, ~11x slower, max spiked to 180ms). Root cause: dynamic quantization's per-call quantize/dequantize overhead (extra kernel launches) dominates when the underlying `Linear` layers are this tiny (dims ~32) — worse penalty than `torch.compile`/autocast since quantization has heavier per-call machinery. Kept disabled.
- [x] **ONNX export** — export to ONNX, benchmark via ONNX Runtime. **Winning technique**: mean ≈ 2.93ms, std ≈ 0.07ms, max ≈ 3.40ms on CPU — beats PyTorch GPU baseline (3.34ms on Modal A10G) and PyTorch MPS baseline (4.86ms), with far more consistent step-to-step timing (much lower std). CPU beat mps here too, consistent with the whole session's theme: this model is too small for "fancier" backends/techniques to pay off — the winning move is removing dispatch/Python overhead (ONNX Runtime's fused graph), not adding optimization machinery on top of PyTorch eager mode.
- [ ] **Final benchmark comparison** — same measurement as stage 1, compare against baseline across all techniques tried.