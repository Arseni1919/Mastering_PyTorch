# my_gpt_lora

A from-scratch implementation of LoRA (Low-Rank Adaptation), applied to a pretrained GPT-2 (124M, loaded from HuggingFace) and fine-tuned on Tiny Shakespeare. The base model's weights come from a pretrained checkpoint and stay frozen throughout — the only thing built and trained here is the LoRA adapter mechanism itself. QLoRA (4-bit quantized base) comes later, once plain LoRA works end-to-end.

## 0. Config

- `model_name` (**'gpt2'**) — 124M parameter pretrained checkpoint
- `lora_rank` (**8**) — the "r" in LoRA; width of the low-rank bottleneck
- `lora_alpha` (**16**) — scaling numerator; `scaling = lora_alpha / lora_rank`
- `lora_dropout` (**0.05**) — dropout applied inside the adapter path only, not the frozen base path
- `target_modules` (**['c_attn', 'c_proj']**) — which projection layers inside each transformer block get wrapped with LoRA
- `block_size` (**128**) — sequence length per training example
- `batch_size` (**16**)
- `learning_rate` (**1e-4**)
- `n_epochs` (**3**)

**Gotcha:** HuggingFace's GPT-2 implementation uses its own `Conv1D` layer for `c_attn`/`c_proj`, not `nn.Linear`. `Conv1D`'s weight is shaped `(in_features, out_features)` — the transpose of `nn.Linear`'s `(out_features, in_features)`. Every section below that touches a base layer's weight shape needs to know which convention it's dealing with.

## 1. Data — `get_data.py`

```txt
load Tiny Shakespeare's raw text (a single plain-text file)

tokenizer = GPT-2's own pretrained BPE tokenizer
token_ids = tokenizer applied to the entire raw text, as one long sequence

ShakespeareDataset — init(token_ids, block_size):
  store token_ids and block_size

ShakespeareDataset — __len__:
  number of block_size windows that fit in token_ids

ShakespeareDataset — __getitem__(idx):
  slice out a contiguous window of block_size tokens starting at idx
  return that window as input_ids (a single tensor — labels reuse the same tensor, since
  the model shifts internally)
```
- No train/val split needed here — the point is to watch the adapter overfit a tiny style, not to generalize.

**Sanity check:**
```python
dataset = ShakespeareDataset(token_ids, block_size=128)
batch = torch.stack([dataset[i] for i in range(8)])
# expected: batch.shape == (8, 128)
```

## 2. LoRALinear — `[Module]`

```txt
LoRALinear — init(base_layer, r, alpha, dropout):
  base_layer = the original frozen layer (Conv1D or Linear) — store it as-is, freeze its parameters
  work out (in_features, out_features) from base_layer's weight shape — mind section 0's
    Conv1D-vs-Linear orientation gotcha
  A = a learnable matrix, shape (r, in_features), random-initialized (small values)
  B = a learnable matrix, shape (out_features, r), initialized to all zeros
  dropout_layer = dropout(dropout)
  scaling = alpha / r

LoRALinear — forward(x):
  base_out = base_layer(x)                          # frozen path, unchanged
  lora_out = dropout_layer(x) @ A^T @ B^T            # low-rank path
  return base_out + lora_out * scaling
```
- `B` starts at all zeros so the adapter is a true no-op at initialization — `lora_out` is exactly zero until training moves `B` away from zero. This is what lets you drop a fresh LoRA adapter into a pretrained model without disturbing its existing behavior.
- `A` and `B`'s shapes must produce a matrix product compatible with `base_layer`'s own `(in_features, out_features)` mapping — get the transpose wrong and either the matmul shapes fail outright, or (worse) they happen to run but compute something meaningless.

**Sanity check:**
```python
base = nn.Linear(32, 16)
lora = LoRALinear(base, r=4, alpha=8, dropout=0.0)
x = torch.rand(2, 32)
out = lora(x)
# expected: out.shape == (2, 16)
# expected: torch.allclose(out, base(x)) — B starts at zero, so LoRA changes nothing yet
```

## 3. Injecting LoRA — `[function]`

```txt
inject_lora(model, target_modules, r, alpha, dropout):
  freeze every parameter in model
  for each named module in model:
    if the module's name matches one of target_modules:
      wrap it: replace that module (in its parent) with LoRALinear(module, r, alpha, dropout)
  return model
```
- "Replace in its parent" means finding the parent module and reassigning its attribute (`setattr`) to the new `LoRALinear` — not just constructing a `LoRALinear` and discarding it. Walking `named_modules()` alone doesn't let you mutate the model; you need each module's parent and attribute name to actually swap it in.
- Freeze *before* wrapping, not after — `LoRALinear.__init__` already freezes `base_layer`'s params individually, but freezing the whole model first is what guarantees nothing outside `target_modules` accidentally stays trainable.

**Sanity check:**
```python
model = inject_lora(model, target_modules=['c_attn', 'c_proj'], r=8, alpha=16, dropout=0.05)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
# expected: trainable / total is well under 1%
```

## 4. Loading the base model — `[function]`

```txt
load_model(model_name, target_modules, r, alpha, dropout, device):
  model = pretrained GPT2LMHeadModel from model_name
  tokenizer = matching pretrained tokenizer
  model = inject_lora(model, target_modules, r, alpha, dropout)
  move model to device
  return model, tokenizer
```

**Sanity check:**
```python
model, tokenizer = load_model('gpt2', ['c_attn', 'c_proj'], r=8, alpha=16, dropout=0.05, device='cpu')
ids = tokenizer('To be or not', return_tensors='pt').input_ids
out = model(ids).logits
# expected: out.shape == (1, ids.shape[1], vocab_size)
```

## 5. Training — `train.py`

```txt
train_step(input_ids):
  input_ids moved to device
  outputs = model(input_ids, labels=input_ids)      # GPT2LMHeadModel shifts + computes cross-entropy internally
  loss = outputs.loss
  backward, optimizer step
```
- Build the optimizer only over `p for p in model.parameters() if p.requires_grad` — passing every parameter would silently waste memory on frozen-gradient bookkeeping (and on some optimizers, actually error if a frozen param has no grad).
- Log loss every epoch (or every N steps) — with `n_epochs=3` and a small dataset, watch for it dropping quickly; a flat loss usually means the optimizer isn't actually touching the LoRA params.

## 6. Saving / loading LoRA weights only — `[function]`s

```txt
save_lora_weights(model, path):
  state_dict = { name: param for name, param in model.state_dict().items() if 'A' or 'B' appears in the name }
  save that dict to path                              # NOT the full model state_dict

load_lora_weights(model, path):
  lora_state_dict = load from path
  model.load_state_dict(lora_state_dict, strict=False)  # strict=False: the checkpoint only covers A/B, not the frozen base
  return model
```
- The whole point of LoRA checkpoints is that they're tiny (a few MB) compared to the full model (~500MB) — saving the entire `state_dict()` here would defeat that.

**Sanity check:**
```python
save_lora_weights(model, 'lora_weights.pt')
import os
# expected: os.path.getsize('lora_weights.pt') well under 10_000_000 (10MB)
```

## 7. Merging LoRA weights — `[function]`

```txt
merge_lora(lora_linear):                              # operates on one LoRALinear instance
  delta_weight = (lora_linear.B @ lora_linear.A) * lora_linear.scaling
  fold delta_weight into base_layer's weight, respecting the Conv1D-vs-Linear orientation
  return a plain layer (Conv1D or Linear) with the folded weight — no A/B, no extra forward-pass cost
```
- This is an optional inference-time optimization: after merging, generation costs exactly what unmodified GPT-2 costs, since the adapter's extra matmuls are gone. Keep the unmerged model around separately if you still want to train further — merging is one-directional here.

**Sanity check:**
```python
x = torch.rand(2, 32)
lora = LoRALinear(nn.Linear(32, 16), r=4, alpha=8, dropout=0.0)
lora.B.data = torch.rand_like(lora.B)     # simulate some training having happened
merged = merge_lora(lora)
# expected: torch.allclose(merged(x), lora(x), atol=1e-5)
```

## 8. Inference — `run_inference.py`

```txt
generate(model, tokenizer, prompt, max_new_tokens):
  ids = tokenizer(prompt)
  generated = model.generate(ids, max_new_tokens=max_new_tokens)   # no_grad, eval mode
  return tokenizer.decode(generated)

compare:
  base_model, tokenizer = load_model(..., r=lora_rank, ...) with A/B left at their fresh (no-op) init
  lora_model = base_model with trained LoRA weights loaded (section 6)
  print generate(base_model, tokenizer, prompt, ...) side by side with
        generate(lora_model, tokenizer, prompt, ...)
```
- Look for the LoRA output skewing toward Shakespeare-ish vocabulary/rhythm (archaic phrasing, verse-like line breaks) compared to the un-adapted base model on the same prompt — that's the visible signal the adapter actually learned something.

## 9. Deferred to later

- **QLoRA** — quantize the frozen base to 4-bit and keep the LoRA adapters in full precision on top. Requires a quantized base layer (e.g. via `bitsandbytes`) in place of the plain `Conv1D`/`Linear` that `LoRALinear` wraps in section 2 — the low-rank adapter math itself doesn't change.
- **LoRA on MLP layers** — extending `target_modules` to also cover GPT-2's MLP `c_fc` projection, not just attention.
- **Rank/alpha sweep** — trying a few different `lora_rank`/`lora_alpha` values to observe the capacity-vs-overfitting tradeoff on a dataset this small.
