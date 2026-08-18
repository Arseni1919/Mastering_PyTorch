# my_leworldmodel

A from-scratch reimplementation of LeWorldModel (LeWM, arXiv:2603.19312) on the Push-T environment only, tuned for fast iteration rather than paper-exact numbers. LeWM is a Joint-Embedding Predictive Architecture that learns a world model end-to-end from raw pixels using exactly **two** loss terms: a next-embedding prediction loss and SIGReg, the anti-collapse regularizer from LeJEPA that pushes embeddings toward an isotropic Gaussian. An encoder maps each frame to a low-dimensional latent; a predictor autoregressively predicts the next frame's latent conditioned on the action. At test time the frozen model is used for planning: Cross-Entropy Method (CEM) searches for an action sequence whose predicted final latent matches a goal image's latent. Everything here — encoder, predictor, SIGReg, the CEM planner — is built from scratch; only the `gym-pusht` environment (and optionally the `lerobot/pusht` dataset) is reused. No async stack, no other environments, no reward, no reconstruction — see section 9.

**The key departure from `my_i_jepa`:** that project kept I-JEPA's stop-gradient on the target branch and used SIGReg only as an *added* anti-collapse term alongside the existing EMA/stop-grad asymmetry. LeWM has **no stop-gradient anywhere** — the prediction target `z_{t+1}` comes from the *same* encoder with gradients flowing, and SIGReg is the *only* thing preventing collapse. This is the whole point of the paper (one regularizer, one hyperparameter, provably non-collapsing), so resist the urge to sneak a `.detach()` onto the target the way I-JEPA did — that would quietly change which method you're implementing.

## 0. Config

- `env_id` (**'gym_pusht/PushT-v0'**) — the registered gym-pusht id
- `img_size` (**96**) — gym-pusht's `obs_type="pixels"` renders natively at 96×96, so no resize step is needed (paper used 224; 96 is the fast choice and happens to be free here)
- `in_channels` (**3**)
- `patch_size` (**16**) — with `img_size=96` gives a 6×6 = 36-patch grid; a `[CLS]` token is prepended (section 2)
- `encoder_embed_dim` (**192**) — ViT-Tiny width, matching the paper's encoder
- `encoder_depth` (**6**), `encoder_num_heads` (**3**) — depth halved from ViT-Tiny's 12 for faster iteration
- `latent_dim` (**128**) — the dimension of the latent space SIGReg and the predictor operate in, *after* the encoder's projection head. Paper uses 192 and §G shows performance drops below ~184; keep 128 while debugging, bump to 192 for a real run
- `predictor_embed_dim` (**192**), `predictor_depth` (**6**), `predictor_num_heads` (**8**)
- `predictor_dropout` (**0.1**) — the paper's §G ablation found a small predictor dropout meaningfully improves downstream control; don't set it to 0
- `frame_skip` (**5**) — one action "block" summarizes 5 raw env steps between kept frames
- `subtraj_len` / `L` (**4**) — kept frames per training window; gives `L-1 = 3` predicted transitions
- `history_len` / `N` (**3**) — predictor's causal context length (paper uses 3 for Push-T)
- `action_dim` / `A` (**2**) — Push-T action is a 2-D target position
- `action_scale` (**256.0**) — raw actions live in [0, 512]; normalize to ≈[-1, 1] as `a / action_scale - 1` (section 1)
- `sigreg_lambda` (**0.1**) — the paper's default and its *only* real hyperparameter; §G shows [0.05, 0.2] all work, 0.5 breaks it
- `sigreg_num_slices` (**256**) — random projection directions; §G shows this barely matters, so keep it small and cheap
- `num_episodes` (**1000**) — subset of the full 20k-episode Push-T dataset; enough to see the latent organize and get a non-trivial success rate
- `batch_size` (**128**), `lr` (**1e-4**), `epochs` (**20**) — more epochs than the paper's 10 to compensate for the smaller dataset
- CEM (section 8): `plan_horizon` / `H` (**5**), `cem_num_samples` (**300**), `cem_iters` (**30**), `cem_num_elites` (**30**), `eval_budget` (**50**), `goal_offset` (**25**)

## 1. Data — `get_data.py`

```txt
PushTWorldModelDataset — init(episodes, frame_skip, subtraj_len, img_size, action_scale):
  store the episodes (each a sequence of frames, actions, and — if available — states)
  precompute a flat list of valid (episode_idx, start_frame) windows:
    a window is valid if start_frame + (subtraj_len - 1) * frame_skip < episode_length
  # doing this once in init keeps __getitem__ O(1) — same precomputed-window idea as
  # my_smolvla's MetaWorldDataset

PushTWorldModelDataset — __len__:
  number of valid windows

PushTWorldModelDataset — __getitem__(idx):
  resolve idx to (episode_idx, start_frame)
  gather L frames at stride frame_skip -> resize to img_size if needed -> to CHW float in [0,1]
  for each of the L-1 transitions, gather ONE action block summarizing the frame_skip raw actions
    between kept frames (see gotcha) and normalize it: a / action_scale - 1
  return obs (L, C, H, W), actions (L, A)   # last action row is unused as input; keep the shape
```
- **Dataset source is a decision, all three work.** (A) Download `lerobot/pusht` — images, actions, and ground-truth states in one place; its states power the linear probe in section 7. (B) The paper's exact DINO-WM Push-T dataset — heavier to obtain, matches paper numbers. (C) Roll your own by running `gym-pusht` under a scripted/noisy policy and logging `(pixels, action)`. Start with (A); reusing `LeRobotDataset` here is the same "don't hand-roll video/parquet plumbing" reasoning used in `my_smolvla`.
- **How to summarize `frame_skip` raw actions into one block:** Push-T's action is an *absolute target position*, so the last raw action of the block is a sensible one-vector summary of "where the pusher is heading." Concatenating all 5 (making `action_dim = 10`) is the faithful alternative. Start with the last action; keep it swappable behind one function so switching later is a one-line change.
- **Normalize actions at the dataset boundary, not scattered through the model.** Raw Push-T actions are in [0, 512]; feeding those into the predictor's AdaLN MLP (section 3) is numerically ugly. Apply `a / action_scale - 1` here, and remember the *inverse* (`(a + 1) * action_scale`) is needed before `env.step` at eval (section 8). Getting this transform inverted is the single most common eval bug.

**Sanity check:**
```python
dataset = PushTWorldModelDataset(episodes, config.frame_skip, config.subtraj_len,
                                 config.img_size, config.action_scale)
obs, actions = dataset[0]
# expected: obs.shape == (config.subtraj_len, config.in_channels, config.img_size, config.img_size)
# expected: actions.shape == (config.subtraj_len, config.action_dim)
# expected: obs.min() >= 0 and obs.max() <= 1, and actions roughly within [-1, 1]
```

## 2. Encoder — `[Module]`

```txt
ViTBlock — init(embed_dim, num_heads):
  # this is the plain, unconditioned ViTBlock you already built in my_i_jepa section 4 —
  # pre-norm, self-attention with no masking, then an MLP. Reuse that design verbatim.

Encoder — init(img_size, patch_size, in_channels, encoder_embed_dim, encoder_depth,
               encoder_num_heads, latent_dim):
  patch_embed = a PatchEmbed (my_i_jepa section 2): non-overlapping patch_size conv into
    encoder_embed_dim, flattened to a token sequence, plus learnable positional embeddings
  cls_token = one learnable vector, shape (1, 1, encoder_embed_dim), prepended to the tokens
  blocks = a container that actually registers submodules, holding encoder_depth ViTBlocks
  proj = a projection HEAD mapping the CLS token into latent space:
    Linear(encoder_embed_dim -> latent_dim) followed by BatchNorm1d(latent_dim)
    # BatchNorm, NOT LayerNorm. Read the gotcha before you type LayerNorm out of habit.

Encoder — forward(o):                                # o: (B, C, H, W)
  tokens = patch_embed(o); prepend cls_token; run through blocks
  cls = the CLS token's output row
  z = proj(cls)                                      # (B, latent_dim)
  return z
```
- **The projection head must use BatchNorm, not LayerNorm — this is load-bearing.** The paper is explicit: a ViT ends in LayerNorm, which normalizes each sample across its features and *prevents SIGReg's anti-collapse objective from being optimized*. SIGReg needs to shape the distribution *across the batch* along each latent dimension; BatchNorm normalizes across the batch per-feature (compatible), LayerNorm normalizes within a sample (orthogonal, and it masks what SIGReg is trying to do). This one substitution is the difference between the method working and silently collapsing.
- The CLS-token-then-projection design is the paper's own encoder recipe. If you'd rather go even faster, a small conv stack ending in global-average-pool + the same BN projection head is a valid swap — the paper's §G shows LeWM is architecture-agnostic (they replaced the ViT with ResNet-18 with no real loss). Keep the BN projector either way; it defines the latent space SIGReg lives in.
- Unlike `my_i_jepa`, there is only ever **one** forward pass of this encoder per frame and **no stop-gradient** on its output — the same `z` tensor feeds the prediction target, the predictor input, and SIGReg, all gradient-connected (section 5).

**Sanity check:**
```python
o = torch.rand(4, config.in_channels, config.img_size, config.img_size)
z = encoder(o)
# expected: z.shape == (4, config.latent_dim)
# expected: z.std(0).mean() is clearly non-zero at init (a healthy spread, not a collapsed point)
```

## 3. Predictor — `[Module]`

```txt
AdaLNPredictorBlock — init(predictor_embed_dim, predictor_num_heads, predictor_dropout, action_dim):
  self_attn = multi-head self-attention over the history axis, CAUSALLY masked
    # token at time t may attend only to times 0..t — temporal causal masking, like a GPT over
    # the sequence of frame latents. This is what makes the predictor autoregressive.
  mlp = expand -> nonlinearity -> project back down, with predictor_dropout
  adaln = an MLP mapping the action to per-block modulation (scale, shift) for the norms —
    the SAME AdaLN modulation you built in my_ltx_video's DiT blocks, conditioned on the ACTION
    here instead of a diffusion timestep
    # ZERO-INIT adaln's final layer so scale->0, shift->0 at start: the predictor ignores the
    # action initially and action-conditioning ramps in gradually. The paper does this explicitly
    # to stabilize training.

AdaLNPredictorBlock — forward(x, a):                 # x: (B, N, predictor_embed_dim), a: (B, N, action_dim)
  produce (scale, shift) from adaln(a)
  x = x + self_attn(modulated_norm(x, scale, shift))   # causal mask applied inside self_attn
  x = x + mlp(modulated_norm(x, scale, shift))
  return x

Predictor — init(latent_dim, predictor_embed_dim, predictor_depth, predictor_num_heads,
                 predictor_dropout, action_dim, history_len):
  in_proj = Linear(latent_dim -> predictor_embed_dim)
  pos_embed = learnable parameter, shape (1, history_len, predictor_embed_dim)
  blocks = a registering container of predictor_depth AdaLNPredictorBlocks
  out_proj = Linear(predictor_embed_dim -> latent_dim) followed by BatchNorm1d(latent_dim)
    # same BN projection-head reasoning as the encoder (section 2)

Predictor — forward(z_seq, a_seq):                   # z_seq: (B, N, latent_dim), a_seq: (B, N, action_dim)
  x = in_proj(z_seq) + pos_embed
  for block in blocks: x = block(x, a_seq)
  return out_proj(x)                                 # (B, N, latent_dim); position t predicts latent t+1
```
- **AdaLN action conditioning, zero-initialized.** You already have the AdaLN machinery from `my_ltx_video`'s DiT — the only change is the conditioning signal is the (normalized) action, not a timestep. The zero-init of the modulation MLP's last layer is not optional polish; the paper calls it out as the thing that keeps early training stable while the predictor learns to actually use the action.
- **Causal masking is over the history/time axis**, exactly analogous to `my_smolvla`'s causal self-attention over the action chunk — here it enforces that predicting latent `t+1` can't peek at latents beyond `t`. This is what lets the predictor be rolled out autoregressively at planning time (section 8).
- No cross-attention here (contrast `my_smolvla`, which cross-attended to VLM features) — the predictor is a pure autoregressive transformer over the latent sequence, with the action injected only through AdaLN.

**Sanity check:**
```python
z_seq = torch.rand(4, config.history_len, config.latent_dim)
a_seq = torch.rand(4, config.history_len, config.action_dim)
z_hat = predictor(z_seq, a_seq)
# expected: z_hat.shape == (4, config.history_len, config.latent_dim)
# with zero-init AdaLN, verify z_hat is (approximately) unchanged when a_seq is replaced by zeros
```

## 4. SIGReg loss — `[function]`

```txt
sigreg_loss(embeddings, num_slices):                 # embeddings: (P, latent_dim) — flatten batch+time axes first
  # This is the SAME Epps-Pulley construction you built in my_i_jepa section 6 — identical
  # random-projection + characteristic-function normality test. Reproduced here for a
  # self-contained file; see that section for the full "why the characteristic function and not
  # moments" rationale.
  directions = num_slices random unit vectors of length latent_dim   # (latent_dim, num_slices)
  t = 17 evaluation points evenly spaced from -5 to 5                 # Epps-Pulley integration grid
  target_cf = exp(-0.5 * t^2)                                         # standard Gaussian's char. function
  projections = embeddings @ directions                              # (P, num_slices), one 1D slice per column
  empirical_cf = mean over P of complex_exponential(i * projections[..., None] * t)   # (num_slices, len(t))
  weighted_sq_error = |empirical_cf - target_cf|^2 * target_cf
  per_slice = P * trapezoidal_integral(weighted_sq_error over t)     # (num_slices,)
  return mean over slices of per_slice                              # scalar
```
- Resample `directions` **fresh every call**, same as `my_i_jepa` — this is what lets a modest 256 slices compound into tight distributional coverage over training rather than fixating on one fixed set of directions.
- **Do NOT standardize `embeddings` before this.** SIGReg's job is to *force* the raw latents toward N(0,1); z-scoring them first removes exactly the signal it's supposed to create. The BatchNorm in the encoder/predictor projection heads gives it a gentle head start on the unit-variance part, but SIGReg does the real shaping — including making the distribution *Gaussian*, not merely unit-variance.
- **Unlike `my_i_jepa`, no separate gradient-connected forward pass is needed to feed SIGReg.** There, the target branch was under `no_grad`, so SIGReg required its own live forward pass to have a gradient path. Here nothing is detached — the encoder latents are already gradient-connected — so you pass those exact latents straight into `sigreg_loss` (section 5). One encoder forward per frame, full stop.

**Sanity check:**
```python
gaussian = torch.randn(512, config.latent_dim)
# expected: sigreg_loss(gaussian, config.sigreg_num_slices) is small — real Gaussians barely fail
collapsed = torch.ones(512, config.latent_dim)
# expected: sigreg_loss(collapsed, config.sigreg_num_slices) is large — the exact failure mode it catches
```

## 5. Full model wiring — `[Module]`

```txt
LeWorldModel — init(...):
  encoder = Encoder(...)
  predictor = Predictor(...)

LeWorldModel — forward(obs, actions):                # obs: (B, L, C, H, W), actions: (B, L, A)
  z = encoder(obs.flatten over B,L).view(B, L, latent_dim)          # encode every frame, ONE pass
  z_hat = predictor(z[:, :-1], actions[:, :-1])                     # predict next latents -> (B, L-1, latent_dim)

  prediction_loss = MSE(z_hat, z[:, 1:])                            # Eq. 1; target is NOT detached
  sigreg = sigreg_loss(z.reshape(-1, latent_dim), sigreg_num_slices)  # regularize ALL latents

  loss = prediction_loss + sigreg_lambda * sigreg
  return loss, prediction_loss, sigreg               # return components for logging
```
- **The prediction target `z[:, 1:]` is deliberately not detached.** This is the defining choice of LeWM versus every EMA/stop-gradient JEPA (including `my_i_jepa`): gradients flow into the encoder through *both* the predicted latents and the target latents. Without SIGReg this would collapse instantly (the encoder would map everything to a constant to zero out the MSE) — which is exactly why SIGReg is non-optional here rather than an add-on. If you find yourself writing `z[:, 1:].detach()`, stop: you've reverted to the I-JEPA recipe.
- SIGReg is applied to the **full block of latents** flattened over batch and time (`(B*L, latent_dim)`), matching the paper's Algorithm 1 (which regularizes the whole `(B, T, d)` tensor).
- Log `prediction_loss` and `sigreg` **separately** — the collapse watch in section 6 depends on watching them independently.

**Sanity check:**
```python
obs, actions = next(iter(loader))          # obs (B,L,C,H,W), actions (B,L,A)
loss, pred_loss, sigreg = model(obs, actions)
# expected: loss.requires_grad is True; pred_loss and sigreg are both positive scalars
loss.backward()
# expected: encoder's parameters all have non-None .grad — SIGReg's gradient must reach the encoder
# (if the BN-vs-LN or standardization gotchas were violated, this is where collapse begins)
```

## 6. Training — `train.py`

```txt
train_step(obs, actions):
  obs, actions to device
  loss, prediction_loss, sigreg = model(obs, actions)
  backward, grad clip (~1.0), optimizer step
```
- Optimizer covers everything (encoder + predictor) — this is a fully end-to-end model, nothing frozen. Use AdamW with a small weight decay.
- **Log `prediction_loss`, `sigreg`, and `z.std(0).mean()` every step — this is your collapse watch.** Healthy: `sigreg` drops sharply early then plateaus (the paper's Fig. 18 shows exactly this), `prediction_loss` decreases steadily, and `z.std` stays healthily non-zero. Collapse looks like `prediction_loss` diving toward 0 *while* `z.std` decays toward 0. If you see that, check the four usual suspects in order: (1) projection head is BatchNorm not LayerNorm; (2) directions are resampled every step; (3) `embeddings` were not standardized before `sigreg_loss`; (4) `z[:, 1:]` was not detached. Then, only if all four are clean, try `sigreg_lambda` up to 0.2.
- Before the full run, overfit a tiny fixed batch (e.g. 8 windows repeated) for a few hundred steps and confirm `prediction_loss` drops sharply — the same "can it memorize a trivial case" check used across this project.

## 7. Linear probe evaluation — `run_probe.py`

```txt
probe(encoder, dataset):                             # dataset must expose ground-truth states
  freeze encoder entirely, no_grad throughout
  for many frames with known state [agent_x, agent_y, block_x, block_y, block_angle]:
    z = encoder(frame)                               # (latent_dim,)
    collect (z, state) pairs
  fit a plain linear least-squares map from z -> state on a train split
    # use numpy/sklearn LinearRegression — don't hand-roll it, same reasoning as reusing
    # sklearn PCA in my_i_jepa
  report per-quantity Pearson correlation r on a held-out split
```
- **This is the gate: do not touch planning until the linear probe passes.** It converts "the loss went down" into "the latent actually encodes the physical state." The paper's Table 1 reports r ≈ 0.97–0.99 for agent and block position on Push-T. If your *linear* probe clears r > 0.9, the world model is real and planning will work; if it's low, the problem is in training (more data, bump `latent_dim` toward 192, train longer) — planning cannot fix a bad encoder.
- gym-pusht's `obs_type="state"` gives exactly `[agent_x, agent_y, block_x, block_y, block_angle]`, so if you took dataset path (A) or generate your own rollouts you get these targets for free — no manual labeling.

**Sanity check:**
```python
r_by_quantity = probe(model.encoder, probe_dataset)
# expected: r for agent_x, agent_y, block_x, block_y are all comfortably > 0.9
# (block_angle is legitimately harder — the paper's own numbers are lower for orientation)
```

## 8. Planning + evaluation — `run_inference.py`

```txt
cem_plan(model, z_start, z_goal, H, num_samples, iters, num_elites):
  mu, var = zeros(H, action_dim), ones(H, action_dim)      # in NORMALIZED action space
  repeat iters times:
    samples = mu + sqrt(var) * standard_normal(num_samples, H, action_dim)
    clip samples to the normalized action bounds (≈[-1, 1])
    # roll each candidate through the FROZEN predictor, autoregressively, from z_start:
    #   z_hat_1 = z_start (tiled to num_samples); z_hat_{k+1} = predictor(z_hat_k, samples[:,k])
    #   BATCH this over all num_samples candidates — 300x5 predictor steps is nothing on GPU
    cost = || z_hat_H - z_goal ||^2   per candidate         # Eq. 4, terminal latent goal-matching
    elites = the num_elites lowest-cost samples
    mu, var = mean and variance of elites along the sample axis
  return mu                                                 # (H, action_dim) plan, normalized

evaluate(model, env, num_episodes):
  freeze model, no_grad throughout
  successes = 0
  for _ in range(num_episodes):
    reset env; pick a goal observation goal_offset steps ahead in a reference trajectory
    z_goal = encoder(goal image)
    steps_used = 0
    while steps_used < eval_budget and not terminated:
      z_start = encoder(current observation)
      plan = cem_plan(model, z_start, z_goal, plan_horizon, cem_num_samples, cem_iters, cem_num_elites)
      for a in plan:                                        # receding-horizon MPC: execute the plan,
        a_env = un-normalize a: (a + 1) * action_scale      # then replan (paper executes full H then replans)
        obs, reward, terminated, truncated, info = env.step(a_env)
        steps_used += 1
        if terminated or steps_used >= eval_budget: break
    successes += 1 if terminated else 0                     # env sets terminated on success (coverage >= 0.95)
  return successes / num_episodes
```
- **Plan in normalized action space, un-normalize only at `env.step`.** The predictor was trained on normalized actions (section 1), so CEM samples and the goal cost live in that space; invert with `(a + 1) * action_scale` right before executing. This is the mirror of `my_smolvla`'s sign-flip warning — the plan can look perfect yet the pusher teleports to a corner if you skip the inverse.
- **Use the env's own success signal.** gym-pusht returns `terminated=True` when block coverage of the goal zone reaches ≥ 0.95, and its reward *is* the coverage. Read success from `terminated`/`info` — don't invent a threshold on latent distance.
- Push-T eval protocol (paper App. F.1): evaluation budget 50 env steps, goal sampled 25 steps ahead, report success rate over ~50 episodes. The paper gets ~90–96%; for this scaled-down setup, a clearly-above-random rate on this real task is the bar to clear, matching numbers is not expected.
- If planning fails *despite* a strong section-7 probe, the suspect is the **predictor rollout, not the encoder**: roll the predictor open-loop for H steps from a real clip and compare each `z_hat_k` against the true `z_k` — the error should grow slowly, not explode. If it explodes, shorten `plan_horizon` or confirm `predictor_dropout` is on.

## 9. Deferred to later

- **Other environments** (TwoRoom, OGBench-Cube, Reacher) — the paper's full evaluation suite. Push-T alone is the fast path to a real result; the rest reuse the identical model and only swap the dataset/env.
- **Full 20k-episode dataset and 224×224 resolution** — the paper's actual scale. Start on the `num_episodes` subset at 96px; scale up once the pipeline clears section 7 and 8.
- **Decoder visualization** (paper Fig. 8) — a small `z -> image` decoder trained *for visualization only* (never backpropped into the encoder), showing the pusher and T-block emerge from a compact latent with no reconstruction loss. A pure interpretability treat.
- **t-SNE of the latent space** (paper Fig. 9) — the 2-D projection should recover the Push-T x–y grid structure, a qualitative confirmation the latent is spatially organized.
- **Violation-of-Expectation** (paper §5.2) — teleport the block mid-trajectory and show the predictor's "surprise" (prediction MSE) spikes; a cheap, striking demo that the model learned physics.
- **The paper's ablation axes** — `latent_dim`, `sigreg_num_slices`, `sigreg_lambda`, predictor size, ViT-vs-CNN encoder. All already set to the paper's best-performing choice here; revisiting them as your own mini-ablations is a natural stretch goal once the base model works.
