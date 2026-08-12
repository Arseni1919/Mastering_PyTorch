# my_i_jepa

A from-scratch reimplementation of I-JEPA (arXiv 2301.08243) on MNIST, with one modernization: I-JEPA's original anti-collapse mechanism (an EMA-updated target encoder + stop-gradient) is replaced with **SIGReg**, a regularizer from LeJEPA (arXiv 2511.08544) that provably prevents representation collapse by pushing embeddings toward an isotropic Gaussian distribution — no momentum scheduling, no fragile hyperparameters. Everything else — multi-block masking, the context/target encoder pair, the predictor conditioned on positional mask tokens, prediction in representation space rather than pixel space — is I-JEPA's own design, unchanged.

**The key architectural change:** a single shared ViT encoder is used for both the context and target forward passes — no separate EMA-updated weight copy. The target forward pass still uses stop-gradient (so gradients only flow through the context→predictor path), and SIGReg is applied to the encoder's target-branch output embeddings, supplying the anti-collapse pressure that used to come from the EMA asymmetry.

## 0. Config

- `patch_size` (**4**) — with `image_size=28`, gives a 7×7 = 49-patch grid
- `image_size` (**28**), `in_channels` (**1**) — MNIST is grayscale
- `embed_dim` (**64**) — encoder's hidden size, small since this is a toy-scale reimplementation
- `encoder_depth` (**6**), `encoder_num_heads` (**4**)
- `predictor_embed_dim` (**32**) — narrower than the encoder, per the paper's own "narrow ViT" predictor
- `predictor_depth` (**4**), `predictor_num_heads` (**4**)
- `num_target_blocks` / `M` (**4**)
- `target_scale_range` (**(0.15, 0.2)**), `target_aspect_ratio_range` (**(0.75, 1.5)**)
- `context_scale_range` (**(0.85, 1.0)**)
- `sigreg_lambda` (**0.05**) — LeJEPA's recommended starting point for λ
- `sigreg_num_slices` (**256**) — random projection directions; smaller than LeJEPA's ImageNet-scale defaults (1024+), appropriate for this toy embedding dimension
- `batch_size` (**64**), `learning_rate` (**1e-3**), `epochs` (**10**)

## 1. Data — `get_data.py`

```txt
load MNIST train/test splits (torchvision has this built in — no need to hand-roll it)
normalize pixel values to a standard range
```
Reusing `torchvision.datasets.MNIST` here is deliberate — same "don't rebuild what a library already does well" reasoning used for tokenizers/datasets elsewhere in this project. The interesting part of this exercise is the JEPA mechanism, not MNIST I/O.

**Sanity check:**
```python
image, label = train_dataset[0]
# expected: image.shape == (1, 28, 28)
```

## 2. Patchify + positional embeddings — `[Module]`

```txt
PatchEmbed — init(image_size, patch_size, in_channels, embed_dim):
  num_patches = (image_size // patch_size) ** 2
  patch_proj = one conv (or equivalent) that turns each non-overlapping patch_size x patch_size
    patch into a single embed_dim vector — kernel_size = stride = patch_size
  pos_embed = a learnable parameter, shape (1, num_patches, embed_dim)

PatchEmbed — forward(x):                          # x: (bs, in_channels, image_size, image_size)
  patches = patch_proj(x), then flatten the spatial grid into one token axis
    # same channels-first -> channels-last reshape care as any patchify step you've built before
  return patches + pos_embed                       # (bs, num_patches, embed_dim)
```

**Sanity check:**
```python
x = torch.rand(2, 1, 28, 28)
tokens = patch_embed(x)
# expected: tokens.shape == (2, 49, config.embed_dim)
```

## 3. Multi-block masking sampler — `[function]`

```txt
sample_block_mask(num_patches_per_side, scale_range, aspect_ratio_range):
  sample a target area = num_patches_per_side^2 * Uniform(*scale_range)
  sample an aspect ratio from aspect_ratio_range
  derive a (height, width) in PATCH units from that area and aspect ratio, clipped to the grid
  sample a random top-left offset so the whole block fits inside the grid
  return the set of patch indices covered by that block

sample_context_and_targets(num_patches_per_side, M, target_scale_range, target_aspect_ratio_range,
                            context_scale_range):
  target_masks = [sample_block_mask(..., target_scale_range, target_aspect_ratio_range) for _ in range(M)]
  context_mask = sample_block_mask(..., context_scale_range, aspect_ratio_range=(1.0, 1.0))
  context_mask = context_mask MINUS the union of all target_masks    # remove overlap
  return context_mask, target_masks
```
- This is one of I-JEPA's two core design choices (the other being predicting in representation space, section 6/7) — the paper's own ablation (their Table 6) shows this specific multi-block strategy beats simpler `random`/`block`/`rasterized` masking by a wide margin, so it's worth getting the overlap-removal step right rather than treating it as an afterthought.
- Removing target overlap from the context is what makes the prediction task non-trivial — without it, the context could trivially "contain" parts of what it's supposed to predict.
- Sample a **fresh** context/target split for every image in every batch, not once per dataset — this is a training-time augmentation, not a preprocessing step.

**Sanity check:**
```python
context_mask, target_masks = sample_context_and_targets(7, config.num_target_blocks,
    config.target_scale_range, config.target_aspect_ratio_range, config.context_scale_range)
# expected: len(target_masks) == config.num_target_blocks
# expected: context_mask has no indices in common with any target_masks[i]
```

## 4. ViT Encoder — `[Module]`

```txt
ViTBlock — init(embed_dim, num_heads):
  norm1 = a normalization layer, size embed_dim
  self_attn = a multi-head self-attention block: embed_dim, num_heads — every token attends to
    every other token, no masking
  norm2 = a normalization layer, size embed_dim
  mlp = expand -> nonlinearity -> project back down, same shape in and out

ViTBlock — forward(x):                             # x: (bs, num_tokens, embed_dim)
  x = x + self_attn(norm1(x))
  x = x + mlp(norm2(x))
  return x
```
```txt
Encoder — init(image_size, patch_size, in_channels, embed_dim, depth, num_heads):
  patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
  blocks = a container that actually registers submodules, holding depth ViTBlock instances

Encoder — forward(x, patch_indices=None):          # x: (bs, in_channels, image_size, image_size)
  tokens = patch_embed(x)
  if patch_indices is not None:
    tokens = tokens indexed along the TOKEN axis, keeping only the entries at patch_indices
      # this shortens the sequence actually fed into the blocks below — the whole reason the
      # context pass is cheaper than a full forward pass, not just a masked-out full-length one
  run tokens through each block in blocks, in order
  return tokens
```
- Pre-norm order (normalize before each sub-layer, not after) — better gradient flow, the standard modern convention.
- No conditioning signal anywhere in `ViTBlock` — no timestep, no class label, nothing like a modulated/adaptive normalization. Just plain self-attention + MLP.
- **One `Encoder` instance, shared weights for both the context and target forward passes** — this is the departure from the original paper's EMA-updated target encoder. Call it once on the masked context patches (gradients flow), and once on the full, unmasked image (wrapped in stop-gradient — see section 7).
- `patch_indices` indexing operates along the token axis (dimension 1, given `(bs, num_tokens, embed_dim)`) — a gather/fancy-index by the actual index values from `sample_context_and_targets`, not a boolean mask over the full grid.

**Sanity check:**
```python
x = torch.rand(2, 1, 28, 28)
full_out = encoder(x)
# expected: full_out.shape == (2, 49, config.embed_dim)
context_out = encoder(x, patch_indices=list(range(10)))
# expected: context_out.shape == (2, 10, config.embed_dim)
```

## 5. Predictor — `[Module]`

```txt
Predictor — init(embed_dim, predictor_embed_dim, predictor_depth, predictor_num_heads, num_patches):
  in_proj = linear, embed_dim -> predictor_embed_dim
  mask_token = one learnable vector, shape (predictor_embed_dim,) — shared across all target positions
  pos_embed = a learnable parameter, shape (1, num_patches, predictor_embed_dim)
  blocks = a registering container of predictor_depth ViTBlocks (predictor_embed_dim, predictor_num_heads)
  out_proj = linear, predictor_embed_dim -> embed_dim

Predictor — forward(context_repr, target_patch_indices):    # context_repr: (bs, num_context_patches, embed_dim)
  context_tokens = in_proj(context_repr)
  mask_tokens = mask_token repeated once per entry in target_patch_indices, batch-expanded, plus
    pos_embed sliced at target_patch_indices                # gives each mask token a position identity
  tokens = concat(context_tokens, mask_tokens) along the sequence axis
  run tokens through blocks
  predicted = out_proj(the slice of tokens corresponding to the mask-token positions)
  return predicted                                    # (bs, len(target_patch_indices), embed_dim)
```
- Called once **per target block** (section 7) — each call conditions on a different set of positional mask tokens corresponding to that block's patch locations, per the paper's description ("we apply our predictor M times").
- The mask tokens carry positional information (via `pos_embed`) but no content information at all — that's the whole point, the predictor has to infer content purely from the context plus "where" it's being asked about.

**Sanity check:**
```python
context_repr = torch.rand(2, 10, config.embed_dim)
target_indices = [3, 8, 15, 22]
predicted = predictor(context_repr, target_indices)
# expected: predicted.shape == (2, 4, config.embed_dim)
```

## 6. SIGReg loss — `[function]`

```txt
sigreg_loss(embeddings, num_slices):                # embeddings: (N, embed_dim) — flatten any batch/token axes first
  directions = num_slices random vectors of length embed_dim, each L2-normalized to unit norm
    # shape (embed_dim, num_slices), so "embeddings @ directions" below projects onto all of
    # them at once

  t = 17 evaluation points, evenly spaced from -5 to 5      # the Epps-Pulley test's integration grid
  target_cf = exp(-0.5 * t^2)                        # standard Gaussian's true characteristic
    # function, evaluated at each point in t — this is the fixed "target" the data gets compared against

  projections = embeddings @ directions               # (N, num_slices) — one 1D "slice" per column
  phase = projections, with a new trailing axis, multiplied against t
    # broadcasts (N, num_slices, 1) against (len(t),) -> (N, num_slices, len(t))
  empirical_cf = mean over N of complex_exponential(i * phase)
    # (num_slices, len(t)), complex-valued — the empirical characteristic function of each slice,
    # evaluated at every point in t

  weighted_sq_error = |empirical_cf - target_cf|^2 * target_cf
    # (num_slices, len(t)) — squared distance between empirical and target CF, weighted by the
    # target CF itself (naturally downweights large |t|, where it's near zero anyway)
  per_slice_statistic = N * (trapezoidal-rule integral of weighted_sq_error over t)
    # (num_slices,) — one Epps-Pulley statistic per direction

  return mean of per_slice_statistic over all directions      # scalar
```
- The core idea, independent of the formula's details: if the embedding distribution really is an isotropic Gaussian, *every* 1D projection of it is also Gaussian (a defining property of multivariate Gaussians) — so testing many random projections is a tractable way to test the whole high-dimensional distribution without the curse of dimensionality. The Epps-Pulley statistic is just *how* each individual projection gets tested: by comparing its empirical characteristic function against the standard Gaussian's true one, rather than comparing histograms or moments directly (moments are numerically unstable for this — high-order moments blow up; the characteristic-function route stays bounded, which is why the paper chose it).
- Resample `directions` fresh every call (every training step), not once — this is what lets a modest `num_slices` (256 here) compound into much tighter distributional coverage over the course of training than a fixed set of directions ever could.
- This formula is transcribed directly from the paper's own reference implementation (their Algorithm 1) rather than derived from the prose description, specifically so the constants (17 points, the `[-5, 5]` range, the `exp(-0.5*t^2)` weighting) are exactly right rather than approximated.

**Sanity check:**
```python
gaussian_samples = torch.randn(512, config.embed_dim)
loss_gaussian = sigreg_loss(gaussian_samples, config.sigreg_num_slices)
# expected: loss_gaussian is small — real Gaussian samples should barely fail the test

collapsed_samples = torch.ones(512, config.embed_dim)   # every embedding identical — total collapse
loss_collapsed = sigreg_loss(collapsed_samples, config.sigreg_num_slices)
# expected: loss_collapsed is large — this is exactly the failure mode SIGReg exists to catch
```

## 7. Full model wiring — `[Module]`

```txt
IJEPAModel — init(...):
  encoder = Encoder(...)
  predictor = Predictor(...)

IJEPAModel — forward(x):                            # x: (bs, in_channels, image_size, image_size)
  context_mask, target_masks = sample_context_and_targets(...)     # section 3, fresh every call

  context_repr = encoder(x, patch_indices=context_mask)            # gradients flow

  with no_grad (stop-gradient):
    target_repr_full = encoder(x)                                  # full image, all 49 patches — used
      # ONLY as the fixed prediction target below, never as SIGReg's input (see gotcha)

  target_repr_full_live = encoder(x)                                # SAME full image, SECOND forward
    # pass, WITHOUT no_grad — gradients flow through this one. SIGReg needs its own gradient-connected
    # tensor; see gotcha for why reusing target_repr_full doesn't work.

  predictions = [predictor(context_repr, target_masks[i]) for i in range(M)]
  targets = [target_repr_full[:, target_masks[i], :] for i in range(M)]

  prediction_loss = average over i of MSE(predictions[i], targets[i])   # I-JEPA's original loss
  sigreg = sigreg_loss(flatten target_repr_full_live over its batch/token axes, config.sigreg_num_slices)

  loss = (1 - config.sigreg_lambda) * prediction_loss + config.sigreg_lambda * sigreg
  return loss, prediction_loss, sigreg               # return the components too, for logging
```
- **Why two separate full-image forward passes, both on the same `x`:** `target_repr_full` (under `no_grad`) exists purely to give `prediction_loss` a fixed, non-moving target — same reasoning as I-JEPA's original stop-gradient. But a tensor produced under `no_grad` has no computation graph attached, so anything computed *from* it — including `sigreg`, if you naively reused `target_repr_full` for it — would also end up with no gradient path back into the encoder. SIGReg's entire job is to apply anti-collapse *pressure* on the encoder's weights; a gradient-disconnected SIGReg term gets computed and logged but silently contributes nothing to `loss.backward()`, which lets the encoder collapse completely unopposed. `target_repr_full_live` is a second, ordinary (gradient-enabled) forward pass over the exact same image, used only for `sigreg` — the extra compute cost (one more encoder forward per step) is the price of SIGReg actually doing its job.
- Since the encoder is genuinely shared (not a separate EMA copy), weights get updated through *two* live paths each step: context→predictor→`prediction_loss`, and `target_repr_full_live`→`sigreg`. Both `no_grad(target_repr_full)` and the live `target_repr_full_live` call recompute the same forward pass on the same input — that duplication is intentional, not a mistake to "optimize away" by sharing one tensor between both roles.

**Sanity check:**
```python
x = torch.rand(4, 1, 28, 28)
loss, pred_loss, sigreg = model(x)
# expected: loss.requires_grad is True
# expected: pred_loss and sigreg are both positive scalars

# the gradient-flow check that would have caught the original bug:
loss.backward()
# expected: model.encoder's parameters all have non-None .grad afterward — if sigreg were the
# only thing driving some particular weight and its grad is None, gradient isn't reaching it
```

## 8. Training — `train.py`

```txt
train_step(x):
  x moved to device
  loss, prediction_loss, sigreg = model(x)
  backward, optimizer step
```
- Log `prediction_loss` and `sigreg` separately, not just their combined `loss` — the whole point of tracking SIGReg is to watch it catch (or fail to catch) collapse; a `prediction_loss` that plummets to near-zero while `sigreg` also stays near-zero is healthy, but a `prediction_loss` near zero while `sigreg` is large means something's off (or expected, early in training).
- Before committing to the full `epochs` run: overfit a tiny fixed batch (e.g. 8 repeated images) for a few hundred steps and confirm `prediction_loss` drops sharply. This is the same "can it memorize a trivial case" sanity check worth running before any full training job in this project.

## 9. Linear probe evaluation — `run_eval.py`

```txt
evaluate(encoder, train_loader, test_loader):
  freeze encoder entirely
  for each (image, label) in train_loader:
    features = encoder(image)                        # full image, no masking
    pool features across the token axis (e.g. mean-pool) into one vector per image
  train a plain linear classifier (logistic regression / a single Linear + cross-entropy) on
    (pooled_features, label) pairs from train_loader
  evaluate that classifier's accuracy on test_loader's pooled features
  return test accuracy
```
- This mirrors the paper's own evaluation protocol (linear probing on frozen features) and is what turns "the training loss went down" into an actual, comparable result — MNIST linear-probe accuracy on top of a self-supervised encoder that never saw a single label during pretraining.
- Mean-pooling the 49 patch tokens into one vector is the simplest choice here; the original paper's own linear-eval setup for image classification does something similar with patch-token pooling (no `[CLS]` token in this architecture, since I-JEPA doesn't use one).

## 10. Representation visualization — `run_inference.py`

```txt
collect_features_and_labels(dataset, encoder, num_per_class):
  freeze encoder entirely, no_grad throughout
  for each digit class 0-9:
    pick num_per_class examples of that class from dataset
  for each picked image:
    features = encoder(image)                        # full image, no masking — same as section 9
    pool features across the token axis into one vector per image
  stack every pooled vector into one (num_classes * num_per_class, embed_dim) array,
    and the matching per-image class labels into a (num_classes * num_per_class,) array
  return features, labels
```
```txt
plot_pca(features, labels):
  reduce features to 2 dimensions via PCA
  for each digit class 0-9:
    scatter-plot that class's 2D points only, using a distinct color and a legend entry
      labeled with the digit it represents
  add a legend, title, axis labels
  show or save the figure
```
- PCA itself isn't something to reimplement here — it's a generic, well-established dimensionality-reduction technique unrelated to what this project is teaching (JEPA architecture, SIGReg). Use `sklearn.decomposition.PCA` directly, same "don't rebuild what a library already does well" reasoning as `torchvision.datasets.MNIST` in section 1.
- This is a qualitative sanity check, not a number like section 9's linear-probe accuracy — but a well-trained self-supervised encoder should show visibly *separated* clusters per digit even in a 2D PCA projection, despite never having seen a single label during pretraining. Overlapping, undifferentiated clusters would suggest the encoder didn't learn much (or collapsed) even if section 9's accuracy number looks passable.

**Sanity check:**
```python
features, labels = collect_features_and_labels(test_dataset, encoder, num_per_class=100)
# expected: features.shape == (1000, config.embed_dim)
# expected: labels.shape == (1000,), containing exactly 100 of each digit 0-9
```

## 11. Deferred to later

- **Dropping stop-gradient entirely** — LeJEPA's own ablations show SIGReg alone (no predictor, no teacher-student asymmetry at all) avoids collapse in their DINO-style multi-crop setup. Whether that holds in this masking-based setup too is genuinely untested here — worth trying once the base version above works, not assumed.
- **Full LeJEPA recipe** — multi-crop global/local views instead of masking, no predictor — as a follow-up comparison against this masking-based version.
- **Scaling beyond MNIST** — CIFAR-10/100, then ImageNet-scale, once the toy setup is validated.
- **Predictor visualization** — the original paper's RCDM-based decoding of predictor outputs back to pixel space (their section 8), as an interpretability stretch goal.
