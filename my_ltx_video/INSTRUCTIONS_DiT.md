# my_ltx_video — DiT

A from-scratch spec for the denoising transformer half of `my_ltx_video/`: a class-conditioned 3D Diffusion Transformer trained via rectified flow, operating on the VAE's latent grid (not pixels). Direct extension of `my_DiT/` from a 2D image DiT to a 3D video-latent DiT — most components are unchanged, called out explicitly where they are. Pseudocode only — fill in the real PyTorch yourself.

Conditioning is class-label only for v1 (KTH's 6 action classes, same `AdaLN` recipe `my_DiT` already uses) — no text, no first-frame/image conditioning yet (see section 13).

## 0. Config

- `num_classes` (**6**) — KTH action classes
- `class_emb_size` (**8**)
- `hidden_size` (**96**) — token embedding dim
- `num_heads` (**4**)
- `mlp_ratio` (**4**) — hidden-width multiplier inside each block's MLP
- `num_layers` (**4**) — number of stacked DiT blocks
- `n_epochs_dit` (**100**)
- `num_inference_steps` (**20**)
- `vae_model_name` (**'saved_vae_model'**) — which trained VAE checkpoint to freeze and load

**Constraint:** `hidden_size` must be divisible by `num_heads` (each head gets an equal slice). `head_dim = hidden_size // num_heads` must be divisible by `6` — section 2's 3D RoPE splits it three ways. With the values above: `head_dim = 96/4 = 24`, and `24/6 = 4`.

Latent grid shape (what the DiT actually operates on) is the VAE's output shape, from `INSTRUCTIONS_VAE.md` section 0/13: `(vae_latent_channels, T, H, W)` where `T = num_frames / 2^num_temporal_downsample_stages`, `H = W = side_size / (vae_patch_size * 2^(num_downsample_stages-1))` — with that file's example values, `(8, 4, 8, 8)`.

## 1. Token embedding — `[Module]`

```txt
TokenEmbed — init(vae_latent_channels, hidden_size):
  one linear projection: vae_latent_channels -> hidden_size
  # this is the "1x1x1 patchifier" from the paper's Table 1 — patchify already happened
  # inside the VAE (see INSTRUCTIONS_VAE.md section 4), so this is just a per-token
  # channel projection, not a spatial patch-split like my_DiT's Patchify conv

TokenEmbed — forward(z):                    # z: (bs, vae_latent_channels, T, H, W)
  move the channel axis to the last position and flatten (T, H, W) into one token axis —
  same channels-first -> channels-last reshape trap as my_DiT's Patchify
  run the flattened tokens through the linear projection
  return tokens: (bs, T*H*W, hidden_size)
```
Note which axis varies fastest when you flatten `(T, H, W)` — section 2's 3D RoPE must assign each token's `(t, h, w)` coordinate using that exact same order, or positions and tokens end up mismatched.

**Sanity check:**
```python
z = torch.rand(2, 16, 4, 8, 8)
tokens = token_embed(z)
# expected: tokens.shape == (2, 4*8*8, hidden_size)
```

## 2. 3D Rotary Positional Embeddings (RoPE) — `build_3d_rope` — `[function]`/buffer

```txt
build_3d_rope(T, H, W, head_dim):
  sixth = head_dim // 6

  # Step 1: each token's (t, h, w) grid coordinate, flattened in the SAME order TokenEmbed used
  t_idx, h_idx, w_idx = a 3-way meshgrid over arange(T), arange(H), arange(W), indexing='ij'
  flatten each to (num_tokens,)

  # Step 2: frequency spread, length `sixth` — same decay formula as build_2d_rope's freq_bands
  freq_bands = ...

  # Step 3: outer product -> angle per (token, frequency), one per axis
  t_angles = outer(t_idx, freq_bands)   # (num_tokens, sixth)
  h_angles = outer(h_idx, freq_bands)
  w_angles = outer(w_idx, freq_bands)

  # Step 4: three axis-channels side by side
  half_angles = concat([t_angles, h_angles, w_angles], dim=-1)   # (num_tokens, head_dim // 2)

  # Step 5: mirror it so slot i and slot i+head_dim//2 carry the same angle
  angles = concat([half_angles, half_angles], dim=-1)   # (num_tokens, head_dim)

  # Step 6: convert angles to the actual rotation components
  return cos(angles), sin(angles)
```
Direct extension of `build_2d_rope` — same mirror-for-rotate-half reasoning applies (see that function's gotcha in `my_DiT/INSTRUCTIONS.md` section 2.2), just split three ways instead of two. `apply_rope` itself (section 6 below) is unchanged from `my_DiT` — `cos`/`sin` just carry three axes' worth of information now instead of two.

**Sanity check:**
```python
cos, sin = build_3d_rope(T=4, H=8, W=8, head_dim=18)
# expected: cos.shape == sin.shape == (4*8*8, 18)
```

## 3. RMSNorm (for QK-normalization) — `[Module]`

```txt
RMSNorm — init(dim):
  one learnable per-channel scale, length dim, initialized to ones

RMSNorm — forward(x):                       # x: (..., dim) — normalizes over the LAST axis only
  divide x by its root-mean-square over that last axis (plus a tiny epsilon, for stability near zero)
  multiply by the learnable scale
  return the result
```
Applied to `q` and `k` independently (separate `RMSNorm` instances — don't share one between them), per attention head, right after they're split into heads and *before* RoPE rotates them — normalize first, then rotate. Normalizing after rotation would distort the rotation's angle-preserving property. This is the paper's enhancement over Pixart-α (section 2.2.2): keeps attention logits from growing unboundedly large as training progresses, which otherwise collapses attention-weight entropy toward one-hot.

**Sanity check:**
```python
x = torch.rand(2, 4, 64, 8)     # (bs, num_heads, num_tokens, head_dim)
out = rms_norm(x)
# expected: out.shape == x.shape; (out**2).mean(dim=-1) ≈ 1 everywhere (scale initialized to ones)
```

## 4. Conditioning vector (timestep + class) — mixed

```txt
sinusoidal_embedding(t, dim):                # [function], pure math, no weights
  half = dim // 2
  frequency spread across `half` bands, decaying geometrically — same formula as
  build_3d_rope's freq_bands
  args = t (scaled up, e.g. *1000, if t is continuous in [0,1)) outer-producted against the frequencies
  return concat(sin(args), cos(args))         # (bs, dim)
```

```txt
TimeEmbedding — init(hidden_size):            # [Module], owns the projection MLP's weights
  a small MLP: hidden_size -> 4*hidden_size -> SiLU -> back to hidden_size

TimeEmbedding — forward(t):
  t_emb = sinusoidal_embedding(t, hidden_size)
  return MLP(t_emb)                           # (bs, hidden_size)
```

```txt
ClassEmbedding — init(num_classes, class_emb_size):                # [Module], nn.Embedding
  one embedding lookup table, num_classes rows, each class_emb_size wide

ClassEmbedding — forward(class_labels):
  return the row for each label: (bs, class_emb_size)
```

```txt
cond = concat(time_embedding(t), class_embedding(class_labels))    # (bs, hidden_size + class_emb_size)
```
`t`-scaling gotcha: since `t` is continuous in `[0, 1)` here (rectified flow), scale it up before feeding it into `sinusoidal_embedding` — unscaled, the frequency bands barely sweep any of their cycle and the embedding collapses toward linear. Keep the *unscaled* `t` for the flow-matching interpolation in `train.py`. Same recipe as `my_DiT`.

**Sanity check:**
```python
t = torch.rand(2)
t_emb = time_embedding(t)
# expected: t_emb.shape == (2, hidden_size)

class_labels = torch.randint(0, num_classes, (2,))
class_emb = class_embedding(class_labels)
# expected: class_emb.shape == (2, class_emb_size)

cond = torch.cat([t_emb, class_emb], dim=-1)
# expected: cond.shape == (2, hidden_size + class_emb_size)
```

## 5. adaLN-Zero modulation — `[Module]`

```txt
AdaLN — init(cond_dim, hidden_size):
  a small MLP: SiLU -> Linear(cond_dim, 6*hidden_size)
  zero-initialize that Linear's weight AND bias

AdaLN — forward(cond):                        # cond: (bs, cond_dim)
  out = MLP(cond)                             # (bs, 6*hidden_size)
  split into 6 equal chunks, each (bs, hidden_size):
    shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp
  return these 6 vectors
```
**Gotcha:** zero-initialize this MLP's final layer (weights AND bias). At init this makes every block behave close to identity (scale≈0, gate≈0), which is what makes deep transformers like this trainable from scratch — skipping this init is a well-documented DiT stability trap. One `AdaLN` instance per `DiTBlock` (section 7) plus one smaller variant inside the final layer (section 8) — same composition as `my_DiT`.

**Sanity check:**
```python
cond = torch.rand(2, hidden_size + class_emb_size)
shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = adaln(cond)
# expected: each of the 6 outputs has shape (2, hidden_size)
# expected (fresh init, before any training): every one of them is exactly zero
```

## 6. Self-attention (3D RoPE + QK-norm) — `[Module]`

```txt
MultiHeadAttnBlock — init(hidden_size, num_heads):
  four linear projections, each hidden_size -> hidden_size, no bias: Q, K, V, output_projection
  head_dim = hidden_size // num_heads
  two RMSNorm instances, each sized head_dim — one for q, one for k (don't share one between them)

MultiHeadAttnBlock — forward(x, rope_cos, rope_sin):        # x: (bs, num_tokens, hidden_size)
  q, k, v = Q(x), K(x), V(x)                 # each (bs, num_tokens, hidden_size)
  reshape each to (bs, num_heads, num_tokens, head_dim)
  q = rms_norm_q(q)
  k = rms_norm_k(k)
  q = apply_rope(q, rope_cos, rope_sin)      # apply_rope itself: unchanged from my_DiT section 2.6
  k = apply_rope(k, rope_cos, rope_sin)      # v is NOT rotated, NOT normalized
  attn_weights = softmax(q @ k^T / sqrt(head_dim))
  out = attn_weights @ v
  merge heads back to (bs, num_tokens, hidden_size)
  return output_projection(out)
```
Everything else matches `my_DiT`'s attention block exactly — same reshape-trap gotcha (heads split/merged via `permute`, not bare `.view()`), same broadcast-shape requirement for `cos`/`sin`. The only additions are the two `RMSNorm` calls.

**Sanity check:**
```python
x = torch.rand(2, 4*8*8, hidden_size)          # (bs, num_tokens, hidden_size); num_tokens = T*H*W
rope_cos, rope_sin = build_3d_rope(T=4, H=8, W=8, head_dim=hidden_size // num_heads)
out = attn(x, rope_cos, rope_sin)
# expected: out.shape == x.shape
```

## 7. DiT block — `[Module]`

```txt
DiTBlock — init(cond_dim, hidden_size, num_heads, mlp_ratio):
  adaln = AdaLN(cond_dim, hidden_size)
  attn = MultiHeadAttnBlock(hidden_size, num_heads)
  layer_norm = LayerNorm(hidden_size, elementwise_affine=False)   # affine off — adaLN supplies it instead
  mlp = Linear(hidden_size, hidden_size*mlp_ratio) -> GELU -> Linear(back to hidden_size)

DiTBlock — forward(x, cond, rope_cos, rope_sin):        # x: (bs, num_tokens, hidden_size)
  shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = adaln(cond)

  h = modulate(layer_norm(x), shift_attn, scale_attn)
  x = x + gate_attn.unsqueeze(1) * attn(h, rope_cos, rope_sin)

  h = modulate(layer_norm(x), shift_mlp, scale_mlp)
  x = x + gate_mlp.unsqueeze(1) * mlp(h)

  return x
```
**Gotcha:** use `LayerNorm` with `elementwise_affine=False` here — the adaLN scale/shift you compute above already provides the learnable affine part; a normal LayerNorm's own built-in affine params would be redundant with (and fight against) adaLN's. Composes one `AdaLN`, one attention block (section 6), and one MLP — stacked `num_layers` times, same composition as `my_DiT`.

**Sanity check:**
```python
x = torch.rand(2, 4*8*8, hidden_size)
cond = torch.rand(2, hidden_size + class_emb_size)
rope_cos, rope_sin = build_3d_rope(T=4, H=8, W=8, head_dim=hidden_size // num_heads)
out = dit_block(x, cond, rope_cos, rope_sin)
# expected: out.shape == x.shape — only values change, not shape
```

## 8. Final layer — `[Module]`

```txt
FinalLayer — init(cond_dim, hidden_size, vae_latent_channels):
  mlp = SiLU -> Linear(cond_dim, 2*hidden_size)     # smaller adaLN variant — shift/scale only, no gate
  zero-initialize that Linear's weight AND bias, same reasoning as AdaLN (section 5)
  layer_norm = LayerNorm(hidden_size, elementwise_affine=False)
  linear = Linear(hidden_size, vae_latent_channels)

FinalLayer — forward(x, cond):
  shift, scale = mlp(cond).chunk(2, dim=-1)
  x = modulate(layer_norm(x), shift, scale)
  return linear(x)
```
Only difference from `my_DiT` section 2.8: output width is `vae_latent_channels`, not `patch_size*patch_size*channels` — there's no pixel-space reconstruction here, the VAE decoder owns that job entirely.

**Sanity check:**
```python
x = torch.rand(2, 4*8*8, hidden_size)
cond = torch.rand(2, hidden_size + class_emb_size)
out = final_layer(x, cond)
# expected: out.shape == (2, 4*8*8, vae_latent_channels)
```

## 9. Un-flatten — `[function]`

```txt
unflatten_tokens(tokens, T, H, W):           # tokens: (bs, T*H*W, vae_latent_channels)
  reshape the token axis back into separate (T, H, W) axes, in the SAME order TokenEmbed flattened them
  move the channel axis back to its original (channels-first) position
  return z_hat: (bs, vae_latent_channels, T, H, W)
```
Pure inverse of section 1's flatten — same axis-order care applies, no learnable parameters.

**Sanity check:**
```python
tokens = torch.rand(2, 4*8*8, 16)
z_hat = unflatten_tokens(tokens, T=4, H=8, W=8)
# expected: z_hat.shape == (2, 16, 4, 8, 8)
```

## 10. Full model wiring — `[Module]`

```txt
DiTFullModel — init(vae_latent_channels, hidden_size, num_heads, num_layers, mlp_ratio, num_classes, class_emb_size, T, H, W):
  cond_dim = hidden_size + class_emb_size

  token_embed = TokenEmbed(vae_latent_channels, hidden_size)
  time_embedding = TimeEmbedding(hidden_size)
  class_embedding = ClassEmbedding(num_classes, class_emb_size)

  rope_cos, rope_sin = build_3d_rope(T, H, W, head_dim=hidden_size // num_heads)   # computed once
    from the latent grid shape (section 0's formula), reused every forward call — non-learnable but
    device-dependent, so register as buffers (not parameters), so they move with .to(device) automatically

  blocks = a container that actually registers submodules, holding num_layers DiTBlock instances —
    each one DiTBlock(cond_dim, hidden_size, num_heads, mlp_ratio)
  final_layer = FinalLayer(cond_dim, hidden_size, vae_latent_channels)

DiTFullModel — forward(z, t, class_labels):             # z: (bs, vae_latent_channels, T, H, W)
  tokens = token_embed(z)
  cond = concat(time_embedding(t), class_embedding(class_labels))

  for block in blocks:                       # num_layers identical DiT blocks
    tokens = block(tokens, cond, rope_cos, rope_sin)

  tokens = final_layer(tokens, cond)
  return unflatten_tokens(tokens, T, H, W)    # same shape as input z
```
Composes every earlier section into the top-level model: `token_embed` (§1) turns the latent grid into tokens, `time_embedding`+`class_embedding` (§4) build the conditioning vector, `blocks` (§7, stacked `num_layers` times) do the actual transformer work, `final_layer` (§8) projects back down, `unflatten_tokens` (§9) restores the grid shape. `rope_cos`/`rope_sin` (§2) are the only pieces of state shared identically across every block — computed once here, passed through unchanged on every call.

**Sanity check:**
```python
z = torch.rand(2, vae_latent_channels, 4, 8, 8)
t = torch.rand(2)
class_labels = torch.randint(0, num_classes, (2,))
out = net(z, t, class_labels)
# expected: out.shape == z.shape — the DiT predicts a velocity field over the same latent grid it consumed
```

## 11. Training — `train_dit.py`

```txt
load the trained VAE encoder — weights from vae_model_name, set to eval(), frozen (no
   gradients ever flow into it: this DiT trains against a fixed latent space)

device, dataloader, net, optimizer (Adam), loss_fn (MSE)

for epoch in range(n_epochs_dit):
  for x, class_labels in dataloader:
    x = normalized to [-1,1], moved to device
    with no_grad: mu, logvar = vae_encoder(x)
    z_1 = mu                                  # deterministic clean latent target — NOT
                                               # reparameterize(mu, logvar); sampling the VAE's
                                               # own noise here would blur the flow-matching target
    t = Uniform(0, 1), one independent value per sample, shape (bs,)
    reshaped_t = t reshaped to broadcast against z_1's (bs, C, T, H, W) shape
    z_0 = standard Gaussian noise, same shape as z_1
    z_t = (1 - reshaped_t) * z_0 + reshaped_t * z_1     # same convention as my_DiT: t=0 -> noise, t=1 -> data
    target = z_1 - z_0                         # velocity
    pred = net(z_t, t, class_labels)           # t stays (bs,) here, NOT reshaped_t
    loss = MSE(pred, target)
    backward + optimizer step
  log epoch average loss
```
Same two-shapes-of-`t` gotcha as `my_DiT` section 3 — independent per-sample `t`, and two different shapes (`(bs,1,1,1,1)` for the interpolation broadcast, `(bs,)` for the model call) needed simultaneously.

## 12. Inference — `run_inference_dit.py`

```txt
load the trained VAE decoder (eval, frozen) and the trained DiT (eval, frozen)

z = pure Gaussian noise, shape (num_samples, vae_latent_channels, T, H, W)   # T,H,W from section 0
class_labels = desired KTH action classes
t = 0.0
step_size = 1 / num_inference_steps

for _ in range(num_inference_steps):
  t_tensor = a (bs,)-shaped tensor filled with the current t, on device
  pred = net(z, t_tensor, class_labels)        # no_grad
  z = z + step_size * pred
  t += step_size

x_hat = vae_decoder(z)                          # no_grad — straight to pixels, no denoising-decoder trick yet
display x_hat frame by frame, same as the VAE's prior-sampling check
```
No scheduler object, same as `my_DiT` — plain inline Euler integration. The full pipeline: noise → DiT (in latent space, class-conditioned) → VAE decoder (to pixels).

## 13. Deferred to later

- **First-frame / image conditioning** — the paper's actual trick (section 2.4): a per-token diffusion timestep instead of one global `t`, so the tokens belonging to the first frame can sit at `t≈0` (barely noised — a conditioning signal) while the rest sit at `t=1`. `image_cond_prob` is already in `config.py` for this. Requires reworking section 4's conditioning and sections 11/12's loops to carry a `(bs, num_tokens)` timestep tensor instead of `(bs,)`.
- **Text conditioning via cross-attention** — the paper conditions on T5 text embeddings through a second attention block per `DiTBlock` (Table 1: "Self + Cross"). Not needed for KTH's class-labeled data.
- **Holistic denoising decoder** — VAE decoder performing the final denoising step (paper section 2.1.1), already noted as deferred in `INSTRUCTIONS_VAE.md`.
- **Timestep-shifted sampling** — the paper samples training `t` from a shifted log-normal distribution rather than uniform, to spend more steps at the timesteps where velocity-prediction is hardest, and shifts further as token count grows (section 2.5.2). Uniform is fine to start.
