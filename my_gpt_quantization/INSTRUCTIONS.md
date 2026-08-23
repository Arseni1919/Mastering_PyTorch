# my_gpt_quant

Compares a pretrained GPT-2 (124M) against a dynamically int8-quantized version of the same weights — no fine-tuning, no LoRA. The point is to see what quantization costs/saves on its own: memory footprint, generation speed, and output quality.

## 0. Config

- `model_name` (**'gpt2'**)
- `prompt` (e.g. **'First Citizen:'**)
- `max_new_tokens` (**100**)

## 1. Load the base model — `[function]`

```txt
load_base_model(model_name):
  model = pretrained GPT2LMHeadModel from model_name, eval mode
  tokenizer = matching pretrained tokenizer
  return model, tokenizer
```

## 2. Convert Conv1D to nn.Linear — `[function]`

```txt
conv1d_to_linear(conv1d):
  read (in_features, out_features) directly off conv1d's weight shape — Conv1D stores it as
    (in, out), the transpose of nn.Linear's (out, in)
  build a fresh nn.Linear(in_features, out_features, bias=(conv1d has a bias))
  copy conv1d's weight into it TRANSPOSED, and its bias unchanged
  return the new nn.Linear

replace_conv1d_with_linear(model):
  walk model's named modules
  for each one that's a Conv1D: swap it (via its parent) for the nn.Linear conversion above —
    same parent-lookup / setattr mechanic as my_gpt_lora's inject_lora
  return model
```
- GPT-2's HuggingFace implementation defines its attention/MLP projections (`c_attn`, `c_proj`, `c_fc`) using its own `Conv1D` class, not `nn.Linear` — a legacy quirk carried over from the original TensorFlow port. This matters a lot for section 3: `torchao`'s quantization functions only recognize `nn.Linear` by default, so `Conv1D` layers are silently skipped — "quantizing" a raw GPT-2 without this conversion step first doesn't actually quantize any transformer-block weights at all.
- Verified directly quantizing a `Conv1D` (skipping this conversion) still lets `torchao` wrap its weight, but breaks at inference: `Conv1D.forward()` calls `torch.addmm`, which `torchao`'s quantized tensor type doesn't implement a dispatch for — `NotImplementedError` at generation time, not at quantization time.

**Sanity check:** converting must not change the model's output at all — it's the same weights, just a different container.
```python
import copy
ids = torch.randint(0, 1000, (1, 5))
before = model(ids).logits
converted = replace_conv1d_with_linear(copy.deepcopy(model))
after = converted(ids).logits
# expected: torch.allclose(before, after, atol=1e-4)
```

## 3. Quantize — `[function]`

```txt
quantize_model(model):
  build a fresh copy of model (quantizing should not mutate the base model you're comparing against)
  convert that copy's Conv1D layers to nn.Linear (section 2) — required first, or nothing gets touched
  apply dynamic int8 weight-only quantization to every nn.Linear in the copy EXCEPT lm_head, in place
  return the quantized copy
```
- Use `torchao`'s current quantization API (`quantize_`, with an int8 dynamic/weight-only config class) rather than the deprecated `torch.quantization.quantize_dynamic` — `torchao` is the actively maintained replacement.
- "Dynamic" means activations are quantized on the fly per forward pass while weights are quantized once, ahead of time — as opposed to static quantization, which needs a calibration pass over sample data first.
- **Exclude `lm_head` specifically**, via `quantize_`'s `filter_fn` argument. GPT-2 ties its output projection (`lm_head.weight`) to its input embedding (`transformer.wte.weight`) — they share the exact same underlying storage (`data_ptr()` is identical between them on a fresh model). Quantizing `lm_head` replaces that shared tensor with a brand-new, independent one, silently breaking the tie. Since `torch.save` is storage-aware and only serializes shared storage once, breaking the tie means the ~147MB embedding matrix gets written out twice instead of once — verified this makes the "quantized" model end up *larger* than the original overall, even though the one layer that got quantized is individually smaller.

**Sanity check:**
```python
base_model, tokenizer = load_base_model('gpt2')
quantized_model = quantize_model(base_model)
# expected: base_model's weights are still full-precision (untouched) — only quantized_model's are int8
# expected: quantized_model.transformer.wte.weight.data_ptr() == quantized_model.lm_head.weight.data_ptr() — tie still intact
```

## 4. Compare memory footprint — `[function]`

```txt
model_size_bytes(model):
  serialize model's state_dict to an in-memory buffer (e.g. torch.save into an io.BytesIO)
  return the buffer's byte length
```
- Do NOT compute this as `sum(p.numel() * p.element_size() for p in model.parameters())`. Verified this reports *identical* sizes for the quantized and unquantized models — `torchao`'s quantized tensors are wrapped in a tensor subclass (`AffineQuantizedTensor`/`Int8Tensor`) that reports `dtype=torch.float32` and `element_size()=4` at the outer level for compatibility with regular PyTorch code, even though the real int8 data lives nested inside it (plus separate `scale`/`zero_point` tensors). Measuring the actual serialized size sidesteps this entirely, since serialization has to write out whatever the real underlying storage is, regardless of what the wrapper claims about itself.

**Sanity check:**
```python
base_bytes = model_size_bytes(base_model)
quant_bytes = model_size_bytes(quantized_model)
# expected: quant_bytes noticeably smaller than base_bytes, now that transformer-block Linears are
# actually being quantized (verified ~51% smaller: 474.75MB -> 232.18MB)
```

## 5. Compare generation output — `[function]`

```txt
generate(model, tokenizer, prompt, max_new_tokens):
  ids = tokenizer(prompt), as a tensor
  generated = model.generate(ids, max_new_tokens=max_new_tokens)   # no_grad, eval mode
  return tokenizer.decode(generated)
```
Run this against both `base_model` and `quantized_model` on the same `prompt`, print side by side. Expect them to be close but not necessarily identical — quantization introduces small numerical error per layer, which can compound over a long generation into visibly different (though still coherent) token choices.

## 6. Compare generation speed — `[function]`

```txt
time_generation(model, tokenizer, prompt, max_new_tokens):
  record a start time
  run generate(...)
  return elapsed time
```
- Run on CPU for this comparison. Dynamic int8 quantization's speedup comes from cheaper integer matmuls on CPU — on GPU, without dedicated low-precision kernels, a naively-quantized model can be just as slow (sometimes slower, from the extra quantize/dequantize overhead) as the original.
- This is hardware-dependent even on CPU: verified on Apple Silicon (M3) that `torchao`'s int8 kernels currently run *slower* than plain fp32 (Apple's Accelerate-backed BLAS is highly tuned; `torchao`'s int8 path isn't accelerated for ARM here) — int8 speedups are most reliable on x86 CPUs with oneDNN/FBGEMM support, or GPUs with dedicated int8 tensor cores. A slowdown on this kind of hardware is an expected, legitimate finding, not a bug.

**Sanity check:**
```python
base_time = time_generation(base_model, tokenizer, prompt, max_new_tokens)
quant_time = time_generation(quantized_model, tokenizer, prompt, max_new_tokens)
# expected: depends on hardware — see gotcha above
```

## 7. Deferred to later

- 4-bit quantization (`bitsandbytes`) alongside this int8 comparison.
- QLoRA — combining this quantized base with the LoRA adapters from `my_gpt_lora` (already noted as deferred there too).
- Static (calibration-based) quantization, as a contrast to the dynamic approach used here.
