# my_smolvla

A from-scratch reimplementation of SmolVLA's novel piece — the Action Expert — on top of a pretrained, frozen VLM backbone. SmolVLA itself is not trained end-to-end from scratch even by its own authors: it pairs a pretrained vision-language model (SmolVLM-2) with a newly-trained flow-matching transformer (the "Action Expert") that predicts robot action chunks. Here, the VLM is loaded pretrained and frozen (same treatment GPT-2 got in `my_gpt_lora`); the Action Expert, the projectors connecting it to the VLM, and the layer-skipping surgery are built from scratch. Trained and evaluated on a small subset of Meta-World tasks. No async inference stack, no community-dataset pretraining, no cross-embodiment — see section 9.

## 0. Config

- `vlm_model_name` (**'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'**) — smallest public SmolVLM2 checkpoint
- `vlm_num_layers_to_keep` (**N = L/2**, where `L` is the pretrained VLM decoder's total layer count) — discard the top half, per the paper's layer-skipping finding
- `action_expert_hidden_size` (**0.75 × VLM hidden dim**) — per the paper's expert-capacity ablation
- `action_expert_num_layers` (**4**)
- `action_expert_num_heads` (**8**)
- `chunk_size` (**n = 10**) — action chunk length; paper found 10–50 a good balance, 10 chosen here for faster iteration
- `num_flow_matching_steps` (**10**) — inference-time integration steps
- `meta_world_tasks` (a short list, e.g. **['reach-v2', 'button-press-v2', 'door-open-v2', 'drawer-open-v2', 'window-open-v2']**) — start small; scale up once this subset works
- `batch_size` (**32**)
- `learning_rate` (**1e-4**)
- `train_steps` (**5000**) — a fraction of the paper's 200k-step pretraining run, appropriate for a single task subset rather than 481 community datasets

## 1. Data — `get_data.py`

```txt
MetaWorldDataset — init(dataset_name, task_names, chunk_size):
  load the underlying LeRobotDataset for dataset_name (lerobot/metaworld_mt50)
  filter it down to episodes belonging to task_names

MetaWorldDataset — __len__:
  number of valid (episode, start_timestep) windows across the filtered episodes,
  where a window needs chunk_size consecutive actions available

MetaWorldDataset — __getitem__(idx):
  resolve idx to a specific episode and start timestep
  read that timestep's image(s), proprioceptive state, and language instruction
  read the next chunk_size actions starting at that timestep
  return (image, state, instruction, action_chunk)
```
- Reusing `lerobot`'s own `LeRobotDataset` here is a deliberate choice, not a shortcut to feel bad about — this dataset's format interleaves parquet metadata with video-encoded camera streams, and hand-rolling that decode/sync logic is pure data-plumbing unrelated to what makes SmolVLA interesting. Same reasoning as using `transformers` for tokenization elsewhere in this project.

**Sanity check:**
```python
dataset = MetaWorldDataset('lerobot/metaworld_mt50', config.meta_world_tasks, config.chunk_size)
image, state, instruction, action_chunk = dataset[0]
# expected: action_chunk.shape == (config.chunk_size, action_dim)
```

## 2. VLM backbone + layer skipping — `[function]`

```txt
load_vlm(vlm_model_name, vlm_num_layers_to_keep):
  model = pretrained SmolVLM-2 model from vlm_model_name
  processor = matching pretrained processor (handles image preprocessing + tokenization + the
    model's own pixel-shuffle visual-token reduction — nothing to build there, it's inherited)
  freeze every parameter in model
  find the language-decoder's layer list inside model
  truncate that list down to its first vlm_num_layers_to_keep entries — same "slice a list of
    submodules" idea as any layer-truncation you'd do on a plain transformer stack
  return model, processor
```
- Truncating a *pretrained* decoder's layer list is a different operation from initializing a smaller model from scratch — you're discarding already-trained top layers, not skipping their existence entirely. The kept layers' weights are untouched.
- Freeze *before* you inspect/return the model — same ordering lesson as `my_gpt_lora`'s `inject_lora`.

**Sanity check:**
```python
model, processor = load_vlm(config.vlm_model_name, config.vlm_num_layers_to_keep)
# expected: number of decoder layers in model == config.vlm_num_layers_to_keep
# expected: sum(p.requires_grad for p in model.parameters()) == 0
```

## 3. Projectors — `[Module]`s

```txt
StateProjector — init(state_dim, vlm_hidden_dim):
  one linear layer, state_dim -> vlm_hidden_dim

StateProjector — forward(state):                 # state: (bs, state_dim)
  return the linear layer applied to state, with a token/sequence axis of length 1 inserted
    # (bs, 1, vlm_hidden_dim) — one token, to be concatenated as a prefix alongside
    # image/language tokens before entering the VLM
```
```txt
FeatureProjector — init(vlm_hidden_dim, action_expert_hidden_size):
  one linear layer, vlm_hidden_dim -> action_expert_hidden_size

FeatureProjector — forward(vlm_features):         # vlm_features: (bs, num_vlm_tokens, vlm_hidden_dim)
  return the linear layer applied to vlm_features  # (bs, num_vlm_tokens, action_expert_hidden_size)
```
```txt
ActionInProjector — init(action_dim, action_expert_hidden_size):
  one linear layer, action_dim -> action_expert_hidden_size

ActionOutProjector — init(action_expert_hidden_size, action_dim):
  one linear layer, action_expert_hidden_size -> action_dim
```
- Feeding state in as a *prefix token into the VLM* (rather than directly into the Action Expert) is a deliberate choice, not the "obvious" option — the paper's own ablation (their Table 11) found this beats the alternative. It matters because the VLM's self-attention layers then get to relate the state to the image/language tokens before the Action Expert ever sees anything.

**Sanity check:**
```python
state = torch.rand(2, state_dim)
state_token = state_projector(state)
# expected: state_token.shape == (2, 1, vlm_hidden_dim)

vlm_features = torch.rand(2, 64, vlm_hidden_dim)   # 64 = paper's visual-token-per-frame count
projected = feature_projector(vlm_features)
# expected: projected.shape == (2, 64, action_expert_hidden_size)
```

## 4. Action Expert — `[Module]`

```txt
ActionExpertBlock — init(hidden_size, num_heads):
  cross_attn = a multi-head attention block: action tokens as queries, VLM features as keys/values
  self_attn = a multi-head attention block: action tokens attend to each other, CAUSALLY masked
    # causal here means action token i can only attend to action tokens 0..i, matching the
    # paper's finding that this beats bidirectional self-attention within the chunk
  mlp = expand -> nonlinearity -> project back down, same shape in and out
  three of whatever normalization layer you've used elsewhere in this project, one before each
    of the three sub-blocks above (pre-norm)

ActionExpertBlock — forward(action_tokens, vlm_features):
  action_tokens = action_tokens + cross_attn(norm(action_tokens), vlm_features)
  action_tokens = action_tokens + self_attn(norm(action_tokens))       # causal mask applied inside
  action_tokens = action_tokens + mlp(norm(action_tokens))
  return action_tokens
```
```txt
ActionExpert — init(hidden_size, num_heads, num_layers):
  blocks = a container that actually registers submodules, holding num_layers ActionExpertBlock
    instances

ActionExpert — forward(action_tokens, vlm_features):
  for block in blocks:
    action_tokens = block(action_tokens, vlm_features)
  return action_tokens
```
- This mirrors the interleaved cross-/self-attention pattern already built in `my_ltx_video`'s DiT blocks, with one addition: here the cross-attention half is real (attending to the frozen VLM's features), not the same-modality self-attention that DiT used throughout.
- The paper interleaves CA and SA *within every block* (not "half the blocks do CA, half do SA") — their ablation (cross+self vs. either alone) found interleaving clearly best. Keep both sub-blocks in every `ActionExpertBlock`.
- Causal masking on the self-attention half specifically prevents action token `t+5` from leaking information to action token `t` — actions later in the chunk shouldn't influence the model's belief about earlier ones during training.

**Sanity check:**
```python
action_tokens = torch.rand(2, config.chunk_size, config.action_expert_hidden_size)
vlm_features = torch.rand(2, 65, config.action_expert_hidden_size)   # 64 visual + 1 state token
out = action_expert(action_tokens, vlm_features)
# expected: out.shape == action_tokens.shape — only values change, not shape
```

## 5. Flow matching objective — `[function]`

```txt
flow_matching_loss(action_expert, vlm_features, action_chunk):
  tau = sample one value per batch element from a Beta distribution
  reshaped_tau = tau reshaped to broadcast against action_chunk's (bs, chunk_size, action_dim) shape
  noise = standard Gaussian noise, same shape as action_chunk
  noisy_actions = reshaped_tau * action_chunk + (1 - reshaped_tau) * noise
  target_velocity = noise - action_chunk          # NOTE the sign/order — this is the paper's own
    # convention (tau=1 -> clean data), double check against section 3.1's equation before
    # trusting this line blindly
  action_tokens = action_in_projector(noisy_actions)
  predicted_velocity = action_out_projector(action_expert(action_tokens, vlm_features))
  return MSE(predicted_velocity, target_velocity)
```
- This is the same flow-matching shape you'd recognize from `my_ltx_video`'s DiT training step (`t=0` noise, `t=1` data, predict velocity) — the difference here is `tau` is Beta-distributed rather than Uniform(0,1), which the paper states improves training (samples more often near the middle of the interpolation, where the hardest predictions live).
- Two different shapes of `tau` needed simultaneously, same gotcha as any flow-matching training step you've built before: one broadcastable against the full action-chunk tensor for the interpolation, one left alone for whatever the Action Expert itself expects as a conditioning input (if it consumes `tau` directly — decide this when wiring section 6).

**Sanity check:**
```python
loss = flow_matching_loss(action_expert, vlm_features, action_chunk)
# expected: loss is a scalar, loss.requires_grad is True
# expected: after a handful of optimizer steps on one fixed (vlm_features, action_chunk) pair,
# loss decreases noticeably — the model should be able to overfit a single batch
```

## 6. Full model wiring — `[Module]`

```txt
SmolVLAModel — init(vlm_model_name, vlm_num_layers_to_keep, state_dim, action_dim,
                     action_expert_hidden_size, action_expert_num_heads, action_expert_num_layers):
  vlm, processor = load_vlm(vlm_model_name, vlm_num_layers_to_keep)
  vlm_hidden_dim = read off vlm's config
  state_projector = StateProjector(state_dim, vlm_hidden_dim)
  feature_projector = FeatureProjector(vlm_hidden_dim, action_expert_hidden_size)
  action_in_projector = ActionInProjector(action_dim, action_expert_hidden_size)
  action_out_projector = ActionOutProjector(action_expert_hidden_size, action_dim)
  action_expert = ActionExpert(action_expert_hidden_size, action_expert_num_heads, action_expert_num_layers)

SmolVLAModel — encode_observation(image, state, instruction):   # everything up to the VLM
  state_token = state_projector(state)
  process image + instruction through processor into the VLM's expected token format
  concatenate state_token as a PREFIX alongside the image/language tokens (section 3's gotcha)
  vlm_hidden_states = run the concatenated tokens through the frozen, truncated vlm
  return feature_projector(vlm_hidden_states)
```
Everything past `encode_observation` (turning `vlm_features` + noisy actions into a velocity prediction) is exactly section 5's `flow_matching_loss` at training time, and section 8's sampling loop at inference time — `SmolVLAModel` itself doesn't need its own top-level `forward`, just this shared `encode_observation` plus the pieces sections 5/8 call directly (`action_expert`, `action_in_projector`, `action_out_projector`).

**Sanity check:**
```python
image, state, instruction, action_chunk = dataset[0]
vlm_features = model.encode_observation(image.unsqueeze(0), state.unsqueeze(0), instruction)
# expected: vlm_features.shape == (1, num_vlm_tokens, config.action_expert_hidden_size)
```

## 7. Training — `train.py`

```txt
train_step(image, state, instruction, action_chunk):
  vlm_features = model.encode_observation(image, state, instruction)     # frozen VLM, no grad needed
    through the VLM itself, but gradients DO need to flow through feature_projector
  loss = flow_matching_loss(model.action_expert, vlm_features, action_chunk)
  backward, optimizer step
```
- Optimizer covers only the trainable pieces: `action_expert`, `state_projector`, `feature_projector`, `action_in_projector`, `action_out_projector` — never the frozen `vlm`.
- Even though `vlm`'s parameters are frozen, don't wrap the whole `encode_observation` call in `no_grad()` — gradients still need to flow *through* the VLM's frozen layers to reach `state_projector` and reach back to `feature_projector`'s input, they just don't accumulate *on* the VLM's own weights. Only genuinely detach if you hit a memory wall and confirm the VLM's output doesn't need a gradient path back to the projectors that feed into it.
- Log the loss regularly; with a small task subset and `train_steps=5000` you should see it drop well below its random-init value within the first several hundred steps if the wiring is correct.

## 8. Inference / evaluation rollout — `run_inference.py`

```txt
sample_action_chunk(model, image, state, instruction, num_flow_matching_steps):
  vlm_features = model.encode_observation(image, state, instruction)     # no_grad
  x = standard Gaussian noise, shape (1, chunk_size, action_dim)
  step_size = 1 / num_flow_matching_steps
  tau = 0.0
  for _ in range(num_flow_matching_steps):
    action_tokens = model.action_in_projector(x)
    predicted_velocity = model.action_out_projector(model.action_expert(action_tokens, vlm_features))
    x = x - step_size * predicted_velocity      # mind the sign: target_velocity was (noise - data),
      # so integrating from noise toward data means subtracting, not adding — the opposite of
      # my_ltx_video's DiT convention, which defined velocity the other way around. Don't copy
      # that sign blindly; re-derive it from section 5's target_velocity definition.
    tau += step_size
  return x                                       # the final x is the sampled action chunk

evaluate(model, task_name, num_trials):
  env = the metaworld environment for task_name
  successes = 0
  for _ in range(num_trials):
    obs = env.reset()
    done = False
    while not done:
      image, state, instruction = extract these from obs (instruction is fixed per task)
      action_chunk = sample_action_chunk(model, image, state, instruction, config.num_flow_matching_steps)
      for action in action_chunk:                # execute the WHOLE chunk before replanning —
        obs, reward, done, info = env.step(action)  # this is "synchronous" inference (section 3.3
        if done: break                               # of the paper, deferred: no async decoupling here)
    successes += info['success']
  return successes / num_trials
```
- This is the step that actually delivers "real results in a simulation env" — run it per task in `meta_world_tasks`, report success rate, compare against the paper's own Meta-World numbers (Table 2) as a sanity check on whether the scaled-down setup is in a remotely reasonable ballpark. Don't expect to match their numbers — smaller data, fewer training steps, smaller task subset — but a non-zero, above-random success rate on at least the easiest tasks is the real bar to clear.

## 9. Deferred to later

- **Async inference stack** (paper section 3.3, Algorithm 1) — the PolicyServer/RobotClient decoupling that overlaps action execution with the next chunk's prediction. Orthogonal to the core modeling work here; synchronous inference (section 8) is simpler and sufficient for a first real result.
- **Full MT50 task coverage** — start with `meta_world_tasks`' small subset, scale up once it's working.
- **Community-dataset pretraining** — the paper's 481-dataset, ~30k-episode pretraining run. Out of reach on single-GPU/CPU hardware and not necessary for a single-benchmark result; training directly on `lerobot/metaworld_mt50` skips this entirely.
- **Cross-embodiment / real-robot transfer** (SO-100, SO-101) — simulation only here.
- **The paper's ablation axes** — chunk size, expert width, layer-skip strategy, regression-vs-flow-matching objective, state-as-prefix-vs-suffix. All already decided by adopting the paper's own best-performing choice at each axis; revisiting them as your own mini-ablations is a natural stretch goal once the base model works.
