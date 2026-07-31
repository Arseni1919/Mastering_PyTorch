# my_dit — Rebuild Instructions

A from-scratch spec for `my_dit/`: a Diffusion Transformer (DiT) trained via flow matching, on raw MNIST pixels. Pure PyTorch — no external transformer/attention libraries. Pseudocode only — fill in the real PyTorch yourself.

## 0. Config

Fields you'll need: `num_classes`, `class_emb_size`, `time_emb_size`, `patch_size`, `hidden_size` (token embedding dim), `num_heads`, `num_layers` (transformer blocks), `mlp_ratio` (hidden width multiplier inside each block's MLP), `side_size`, `n_epochs`, `batch_size`, `learning_rate`, `num_inference_steps`.

**Constraint:** `side_size` must be divisible by `patch_size` (patchify needs to tile the image exactly, no leftover pixels). `hidden_size` must be divisible by `num_heads` (each head gets an equal slice of the embedding).

## 1. Data (`get_data.py`)

Same as `my_ddpm`: MNIST, padded to `side_size`, converted to tensor.

## 2. Model (`define_model.py`) — main focus

**Module vs. function rule:** if a piece owns learnable parameters (weights/biases — anything that needs to train, show up in `state_dict()`, move with `.to(device)`, or be seen by the optimizer), it must be an `nn.Module`. If it's pure computation/reshaping with nothing to learn, a plain function is simpler and sufficient — don't wrap it in a `Module` just for structure. A third category shows up in 2.2 below: non-learnable but *device-dependent* precomputed state (no gradient, but still a tensor that must live on the right device) — register it with `register_buffer` if it's part of a Module (so `.to(device)` moves it automatically), or just recompute it fresh each forward call if it's cheap (fine at MNIST scale). Each subsection below is tagged accordingly.

### 2.1 Patchify — `[Module]` (owns the projection conv's weights)
```txt
patchify(x):                            # x: (bs, ch, side, side)
  split into non-overlapping patch_size x patch_size patches, project each to hidden_size
  # standard trick: a single conv with kernel_size=stride=patch_size does patch-split + projection in one op
  return tokens: (bs, num_patches, hidden_size)     # num_patches = (side/patch_size)^2
```
**Gotcha:** the conv output comes out as `(bs, hidden_size, side/patch_size, side/patch_size)` — you still need to flatten the two spatial dims into one and move `hidden_size` to the last axis (same channels-first → channels-last reshape trap as the UNet's attention block: `.view()` alone scrambles it, you need to actually move the axis before flattening). No positional info is added here — that now happens inside attention (2.2/2.6), not on the token stream.

### 2.2 2D Rotary Positional Embeddings (RoPE) — `[function]` / buffer, current best practice
```python
def build_2d_rope(num_patches_per_side, head_dim, device):
    quarter = head_dim // 4

    # Step 1: each patch's (row, col) grid coordinate, flattened row-major
    #   torch.meshgrid(indexing='ij') on two arange(...) gives you 2D row/col grids;
    #   .reshape(-1) flattens each into (num_patches,) — must match patchify's flatten order
    row_idx, col_idx = torch.meshgrid(torch.arange(___), torch.arange(___), indexing='ij')
    row_idx = row_idx.reshape(-1)___   # (num_patches,) — don't forget device here
    col_idx = col_idx.reshape(-1)___   # (num_patches,) — same

    # Step 2: frequency spread, length `quarter` — same decay formula as sinusoidal_embedding's freqs
    freq_bands = ___   # (quarter,), moved to device

    # Step 3: outer product -> angle per (patch, frequency)
    row_angles = torch.outer(___, ___)   # (num_patches, quarter)
    col_angles = torch.outer(___, ___)   # (num_patches, quarter)

    # Step 4: row-channels + col-channels side by side
    half_angles = torch.cat([___, ___], dim=-1)   # (num_patches, head_dim // 2)

    # Step 5: mirror it so slot i and slot i+head_dim//2 carry the same angle
    angles = torch.cat([___, ___], dim=-1)   # (num_patches, head_dim)

    # Step 6: convert angles to the actual rotation components
    return ___, ___   # cos, sin — each (num_patches, head_dim), rotate q/k, not added to tokens
```
**Gotcha:** `torch.outer` needs both its inputs on the same device — `row_idx`/`col_idx` from `torch.meshgrid`/`torch.arange` default to CPU, so if you only move `freq_bands` to `device` and forget the indices, you'll get a device-mismatch error the moment `device != 'cpu'`.

This replaces the learned absolute positional embedding some earlier vision transformers use — no separate parameter table, and it generalizes better to grid sizes not seen during training since it encodes *relative* offset directly into the attention dot product rather than an absolute per-position vector. Original DiT paper used fixed sinusoidal absolute embeddings; RoPE is the more modern choice used in several newer ViT/DiT variants. Compute once (not per-block — every block shares the same table, since patch positions don't change) and pass into each block's attention call.
**Gotcha:** the final duplication isn't decorative. `apply_rope` (2.6) pairs dimension `i` with dimension `i + head_dim/2` via rotate-half — for that to be a genuine rotation (not a random linear mix), the angle at `i` and at `i + head_dim/2` must match exactly, i.e. `cos[i] == cos[i+head_dim/2]`. That's *why* the row/col split happens one level down, at `head_dim/4` (via `quarter`), and gets mirrored into both halves — not a flat `head_dim/2` row/col split. This is meaningfully more error-prone to implement than a lookup-and-add embedding. Verify with a small sanity check: rotating `q`/`k` for two identical positions should reduce to the exact unrotated dot product.

### 2.3 Conditioning vector (timestep + class) — mixed
```txt
t_emb = sinusoidal_embedding(t, time_emb_size) -> small MLP        # same idea as my_ddpm
class_emb = embedding lookup for class_labels
cond = t_emb + class_emb                          # combined into ONE vector, shape (bs, hidden_size)
```
`sinusoidal_embedding` itself — `[function]` (pure math, no weights, exactly like in `my_ddpm`). The MLP after it and the class embedding lookup — `[Module]` each (both own weights: `nn.Linear`/`nn.Sequential` and `nn.Embedding` respectively). The final `+` combining them is just an op inside the top-level model's `forward`, not its own component.
**Gotcha carried over from my_ddpm:** if `t` is continuous in `[0, 1)` (flow matching), scale it up (e.g. `t * 1000`) before feeding it to `sinusoidal_embedding` — the frequency bands are tuned for a `~1000`-range input; fed raw `[0,1)` values they barely sweep any of their cycle and the embedding collapses to a near-linear signal. Keep the *unscaled* `t` for anything else (e.g. the training interpolation, done in `train.py`).

### 2.4 adaLN-Zero modulation — `[Module]` (owns the MLP's weights; one instance per DiT block)
```txt
adaLN(cond):                             # cond: (bs, hidden_size)
  out = MLP(SiLU(cond))                  # -> 6 * hidden_size
  split into 6 chunks, each (bs, hidden_size):
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
  return these 6 vectors
```
**Gotcha:** zero-initialize this MLP's final layer (weights AND bias). At init this makes every block behave close to identity (scale≈0, gate≈0), which is what makes deep transformers like this trainable from scratch — skipping this init is a well-documented DiT stability trap.

### 2.5 Modulation helper — `[function]` (pure arithmetic, no weights)
```txt
modulate(x, shift, scale):                # x: (bs, num_patches, hidden_size); shift/scale: (bs, hidden_size)
  return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)     # broadcast over the patch dimension
```

### 2.6 Self-attention — `[Module]` (owns the q/k/v/output projection weights), standard multi-head, from scratch
```txt
apply_rope(x, cos, sin):                  # x: q or k, (bs, num_heads, num_patches, head_dim); cos/sin: (num_patches, head_dim)
  x1, x2 = split x's last dim in half
  rotated = concat(-x2, x1)                # the standard "rotate half" trick
  return x * cos + rotated * sin           # cos/sin broadcast across batch and heads

attn(x, rope_cos, rope_sin):               # x: (bs, num_patches, hidden_size)
  q, k, v = linear projections of x, each (bs, num_patches, hidden_size)
  reshape each to (bs, num_heads, num_patches, head_dim)     # head_dim = hidden_size / num_heads
  q = apply_rope(q, rope_cos, rope_sin)
  k = apply_rope(k, rope_cos, rope_sin)    # v is NOT rotated — only q/k need positional info for the dot product
  attn_weights = softmax(q @ k^T / sqrt(head_dim))            # over num_patches
  out = attn_weights @ v
  merge heads back to (bs, num_patches, hidden_size)
  return output_projection(out)
```
**Gotcha:** same reshape trap as before — splitting into heads and merging back both need axes physically moved (`permute`/`transpose`), not just `.view()`'d, or values land in the wrong head/position. `cos`/`sin` need an extra broadcast dim (`(1, 1, num_patches, head_dim)`) to line up against `q`/`k`'s `(bs, num_heads, num_patches, head_dim)`.

### 2.7 DiT block — `[Module]` (composes the Modules above; stacked num_layers times)
```txt
dit_block(x, cond, rope_cos, rope_sin):
  shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = adaLN(cond)

  h = modulate(layernorm(x), shift_attn, scale_attn)
  x = x + gate_attn.unsqueeze(1) * attn(h, rope_cos, rope_sin)

  h = modulate(layernorm(x), shift_mlp, scale_mlp)
  x = x + gate_mlp.unsqueeze(1) * mlp(h)          # mlp: Linear(hidden, hidden*mlp_ratio) -> GELU -> Linear(back to hidden)

  return x
```
**Gotcha:** use `LayerNorm` with `elementwise_affine=False` here — the adaLN scale/shift you compute above already provides the learnable affine part; a normal LayerNorm's own built-in affine params would be redundant with (and fight against) adaLN's.

### 2.8 Final layer — `[Module]` (owns its own adaLN + linear-projection weights)
```txt
final_layer(x, cond):
  shift, scale = a smaller adaLN variant of cond -> 2 * hidden_size, split in two
  x = modulate(layernorm(x), shift, scale)
  return linear(x)                        # hidden_size -> patch_size*patch_size*channels
```

### 2.9 Unpatchify — `[function]` (pure reshape/permute, no weights)
```txt
unpatchify(tokens):                       # tokens: (bs, num_patches, patch_size*patch_size*channels)
  reshape each token back into its patch_size x patch_size x channels patch
  reassemble the grid of patches into the full (bs, channels, side, side) image
```
**Gotcha:** the inverse of section 2.1's reshape trap — going from "one row per patch" back to "one 2D grid" needs the spatial patch-row/patch-col axes explicitly separated and interleaved with pixel-row/pixel-col via `permute`, not a flat reshape.

### 2.10 Full model wiring — `[Module]` (the top-level model; owns/registers all the Modules above)
```txt
# in __init__: rope_cos, rope_sin = build_2d_rope(side/patch_size, head_dim)   # computed once, reused every forward call

dit_forward(x, t, class_labels):
  tokens = patchify(x)                     # no positional embeddings added here anymore — RoPE lives inside attention
  cond = time_embed(t) + class_embed(class_labels)

  for block in blocks:                    # num_layers identical DiT blocks
    tokens = block(tokens, cond, rope_cos, rope_sin)

  tokens = final_layer(tokens, cond)
  return unpatchify(tokens)                # same shape as input x
```

## 3. Training (`train.py`) — flow matching

```txt
device = pick cuda, else mps, else cpu
dataloader = DataLoader over the dataset
net, optimizer (Adam), loss_fn (MSELoss)

for epoch in range(n_epochs):
  for x, y in dataloader:
    x = x * 2 - 1                                    # normalize to [-1, 1]
    t = Uniform(0, 1), ONE INDEPENDENT VALUE PER SAMPLE, shape (bs,)   # not one shared scalar for the batch
    reshaped_t = t reshaped to (bs, 1, 1, 1)          # for broadcasting against the image
    x_0 = standard Gaussian noise, same shape as x
    x_1 = x                                            # the real image
    x_t = (1 - reshaped_t) * x_0 + reshaped_t * x_1    # straight-line interpolation
    target = x_1 - x_0                                  # velocity
    pred = net(x_t, t, y)                               # t stays (bs,) here, NOT reshaped_t
    loss = MSE(pred, target)
    backward + optimizer step
  log epoch average loss
```
**Gotchas:** `t` must be independent per sample (`torch.rand((bs,))`), not one shared scalar for the whole batch — a shared `t` is a real, if subtle, statistical-efficiency bug (every example in a step sees the same corruption level). Two different shapes of `t` are needed simultaneously: `(bs,1,1,1)` for the interpolation broadcast, `(bs,)` for the model call — don't conflate them.

## 4. Inference (`run_inference.py`) — Euler ODE integration

```txt
x = pure Gaussian noise, shape (num_samples, channels, side, side)
y = desired class labels
t = 0.0
step_size = 1 / num_inference_steps

for _ in range(num_inference_steps):
  t_tensor = a (bs,)-shaped tensor filled with the current t, on device
  pred = net(x, t_tensor, y)              # no_grad
  x = x + step_size * pred
  t += step_size

rescale x from [-1,1] back to [0,1] for display
```
No scheduler object needed — this is plain numerical integration, all inline.

## 5. Improving model performance — standard order of operations

- **Overfit a tiny subset first** — rules out pipeline bugs before blaming capacity.
- **Check the loss curve shape** — decides whether you need capacity, more steps, or optimization fixes.
- **Scale width first** (`hidden_size` — a `head_dim` of ~4 is tiny; try 64-128, divisible by `num_heads`) — cheapest, highest-leverage lever.
- **Scale depth next** (`num_layers`, e.g. 3 → 6) — only pays off once width is reasonable.
- **Increase `n_epochs`** — transformers need more iterations than CNNs to converge from scratch.
- **Tune LR schedule (warmup + decay)** — only if loss curve looks unstable, not undersized.
- **Visually check samples via `run_inference.py`** — final validation; low loss ≠ good samples.
- **Change one dial at a time** — otherwise you can't attribute the improvement.
