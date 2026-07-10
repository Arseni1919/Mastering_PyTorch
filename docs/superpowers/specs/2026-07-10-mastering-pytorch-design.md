# MasteringPytorch — Design

## Goal

Reach expert-level PyTorch by implementing 10 SOTA algorithms from scratch. Every project
produces a real, visible result on a small dataset within **one hour of single-GPU training on
Modal**. After the reference implementation lands, the same project is re-implemented by hand
against a numeric test oracle.

## The From-Scratch Rule

> Write every line of the *architecture* yourself. Loading pretrained weights into your own
> modules, or freezing a pretrained encoder, is allowed wherever the original paper did it.

Rationale: from-scratch + one hour + a result worth looking at is unsatisfiable for anything
that in reality depends on internet-scale pretraining (CLIP, VLM, VLA). The interesting part is
the architecture and the objective; a base model having read the internet is donated, not
re-derived. This is what the LLaVA and π0 papers themselves did.

"Real result" means a nice result on a small dataset — not production scale.

## Curriculum

Each is a self-contained folder, built one at a time. Numbering is a dependency map, not a
schedule: `05` and `06` require `04`, `09` requires `03` and `08`, everything else is free
standing. **Build order starts with `02-ddpm`.**

| # | Project | Written from scratch | ≤1h result |
|---|---------|----------------------|------------|
| `01` | vit | everything | CIFAR-10 ~90%, attention maps |
| `02` | **ddpm** | UNet, noise schedule, DDIM, CFG | class-conditional CIFAR-10 grid |
| `03` | latent-flow-dit | DiT, rectified flow | 128px Oxford Flowers. Frozen SD-VAE |
| `04` | gpt | BPE, RMSNorm, RoPE, SwiGLU, GQA | TinyStories → coherent stories |
| `05` | inference | KV cache, top-p, spec decoding, int8 | tok/s table vs `04`. Zero GPU-hours |
| `06` | moe-ssm | sparse MoE layer, Mamba-2 block | three loss curves vs `04` |
| `07` | grpo | the RL loop, GRPO objective | Qwen3-0.6B on Countdown, reward climbs |
| `08` | vlm | projector, cross-modal plumbing | frozen SigLIP + Qwen3-0.6B → image QA |
| `09` | vla | flow-matching action expert | `08` head-swapped → Push-T rollouts |
| `10` | lewm | ViT encoder, AdaLN predictor, SIGReg | Push-T planning via CEM/MPC |

Topic coverage: diffusion (`02`,`03`), CV (`01`,`10`), LLM/text-gen (`04`–`07`), VLM (`08`),
VLA (`09`), JEPA (`10`).

Deliberately excluded: DETR/detection, CLIP-from-scratch (data-hungry; `08` teaches contrastive
plumbing anyway), VQ-VAE, GNNs, audio.

Perf work (`torch.compile`, AMP, profiler, FSDP) is folded into the projects that need it, not
given its own tier — the concepts stick when there is a real bottleneck to fix.

### Compounding

`04` → `05`, `06`. `03`'s flow matching → `09`'s action expert. `08` → `09`. By `09` you graft
your own modules onto a checkpoint you already modified once.

### The `09`/`10` pairing

Both act in Push-T. That collision is the design, not an accident:

| | `09-vla` | `10-lewm` |
|---|---|---|
| paradigm | model-**free** policy | model-**based** planning |
| learns | π(action \| image, instruction) | latent dynamics; no policy |
| signal | imitation (flow matching) | next-embedding + SIGReg; no reward |
| acts by | one forward pass | CEM/MPC search through the world model |

Same env, same demo data, same eval harness, one shared success-rate number. `09` conditions on
templated instructions ("push the T to the bottom-left") over goal variants, making the *V-L*-A
real rather than nominal.

`10` implements **LeWorldModel** (arXiv 2603.19312, LeCun et al., Mar 2026): ViT-tiny encoder
(5M) + 6-layer transformer predictor (10M), actions injected by zero-init AdaLN. Two losses:
MSE on the next embedding, plus SIGReg — project embeddings onto M=1024 random unit directions
and apply the univariate Epps–Pulley normality test to each 1D marginal (Cramér–Wold: matching
all marginals matches the joint). λ=0.1. No EMA, no stop-gradient, no frozen encoder. Inference
is CEM with 300 sampled action sequences, 30 iterations, top-30 elites, under MPC.

## Repo Architecture

```
MasteringPytorch/
  pyproject.toml          uv, python 3.12, torch 2.13, modal 1.5
  CLAUDE.md               code conventions (enforced, no exceptions)
  <NN>-<name>/
    README.md             the paper, the idea, the expected result
    model.py              reference implementation — one module
    template.py           signature-only stubs — you fill these in
    train.py              Modal entrypoint
    utils.py              this project's helpers and Modal harness
    tests/
      conftest.py         puts the project dir on sys.path
      test_<name>.py      the oracle
    assets/               generated samples, curves (gitignored except finals)
```

Exactly one reference module per project, mirrored by exactly one `template.py`. Splitting the
reference across files would force a matching split of the template and an asymmetric oracle.

**Projects are isolated.** There is no repo-root `utils.py` and no cross-project imports. Each
project carries its own copy of `load_impl`, `seed_everything`, `save_grid`, and the Modal
`app`/`image`/`volume`. This duplicates perhaps thirty lines per project — deliberately. A project
must be readable, runnable, and copy-pasteable on its own, and later projects must be free to
change their harness without disturbing earlier ones. The "put it in `utils.py`" convention scopes
*within* a project, not across the repo.

A function moves to root `utils.py` only once a second file needs it. No speculative shared
layer.

### Modal harness

One `App`, one `Image`, one `Volume` for checkpoints and datasets, defined in `utils.py`.
Training functions declare `gpu=`, `timeout=3600`, and mount the volume. Profile `ars-69317`,
already authenticated at `~/.modal.toml`.

## The Template Oracle

The core of the learning loop. A blank file cannot tell you that you got it right — a subtly
wrong causal mask still trains and still emits a plausible loss curve.

Each project ships `model.py` (reference), `template.py` (stubs), and one test file that runs
against **either**, selected by the `IMPL` environment variable:

```python
IMPL = os.environ.get("IMPL", "template")
Attention = import_module(IMPL).Attention


def test_matches_reference():
    torch.manual_seed(0)
    ref = model.Attention(64, 4)
    mine = Attention(64, 4)
    mine.load_state_dict(ref.state_dict())
    x = torch.randn(2, 16, 64)
    assert_close(mine(x), ref(x))
```

```
IMPL=template uv run pytest ddpm    ->  fails until you implement it
IMPL=model    uv run pytest ddpm    ->  passes
```

**Constraint this imposes:** `load_state_dict` parity requires the template to declare the same
submodule names in the same creation order as the reference. Stubs therefore name the attributes
(`self.qkv`, `self.proj`) in their docstring and leave only the logic blank. This is an accepted
cost — a numeric oracle is worth more than freedom in naming.

Tests cover, in order: parameter count, output shapes, numeric parity against the reference, and
one overfit-a-single-batch run.

## Project `02-ddpm`

The first build.

### Result

A 10×10 class-conditional CIFAR-10 grid (one row per class), a classifier-free-guidance scale
sweep, and a loss curve. Samples recognizable as their class.

### Reference implementation

`model.py` — the UNet and the diffusion process in one module.

- sinusoidal timestep embedding → 2-layer MLP
- class embedding table with a null token for CFG
- `ResBlock`: GroupNorm → SiLU → conv, timestep+class embedding added as a bias, residual
- `Attention`: single-head self-attention at 16×16 resolution
- `UNet`: base 128 channels, mults (1, 2, 2, 2), ~35M params
- `cosine_abar`: cosine noise schedule (improved DDPM)
- `Diffusion.q_sample`: closed-form forward noising
- `Diffusion.p_losses`: predict ε, MSE
- `Diffusion.ddpm_sample`: ancestral, 1000 steps
- `Diffusion.ddim_sample`: deterministic, 50 steps
- classifier-free guidance: null-token dropout at 10% during training, scale `w` at sampling

### Training

CIFAR-10 via torchvision, batch 256, AdamW, bf16 autocast, `torch.compile`, EMA of weights
(non-negotiable for sample quality). One H100, ~50k steps in the hour.

### Verification

`tests/test_ddpm.py` — parity on `ResBlock`, `Attention`, `UNet.forward`, `q_sample`, `p_losses`,
and the DDIM step; a schedule-property test (ᾱ monotonically decreasing, ᾱ₀≈1, ᾱ_T≈0); an
overfit-32-images test.

## Code Conventions

Enforced on every file, no exceptions. Mirrored into `CLAUDE.md`.

- Max 100 characters per line.
- No empty lines inside functions.
- Two empty lines between functions.
- No comments inside code unless the WHY is genuinely non-obvious.
- All imports at the top of the file — never mid-file.
- Functions focused and sharp; split when logic accumulates.
- No duplicated code. Check for reuse before writing.
- A function used in more than one file lives in `utils.py`.
- As little code as possible. Simple and direct over clever and verbose.

## Workflow

For each project, in order:

1. Reference implementation + Modal training run, until the real result exists.
2. Templates and tests derived from the working reference.
3. You re-implement into `template.py` until `IMPL=template uv run pytest` is green.
4. Only then does the next project begin.
