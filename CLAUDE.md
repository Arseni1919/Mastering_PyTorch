# MasteringPytorch

Ten SOTA algorithms implemented from scratch to reach expert-level PyTorch. Each lives in its own
folder and produces a real result on a small dataset within one hour of single-GPU training on
Modal. Design: `docs/superpowers/specs/2026-07-10-mastering-pytorch-design.md`.

## Code Conventions

These apply to all code files in this project, no exceptions.

### Formatting
- Max 100 characters per line.
- No empty lines inside functions.
- Two empty lines between functions.
- No comments inside code unless the WHY is genuinely non-obvious.

### Structure
- All imports at the top of the file — never mid-file.
- Functions must be focused and sharp. If a function has too much logic, split it into
  subfunctions.
- No duplicated code — always check if logic can be reused before writing it again.
- If a function is used in more than one file, put it in `utils.py` and import from there.
- Use as little code as possible — prefer simple and direct over clever and verbose.

## Layout

```
<NN>-<name>/
  README.md             the paper, the idea, the expected result
  model.py              reference implementation
  template.py           signature-only stubs to re-implement by hand
  train.py              Modal entrypoint
  tests/test_<name>.py  numeric oracle, runs against either impl
```

## The Template Oracle

`template.py` must declare the same submodule names in the same creation order as `model.py`, so
`load_state_dict` transfers between them and outputs can be compared numerically.

```
IMPL=template uv run pytest 02-ddpm    # fails until you implement it
IMPL=model    uv run pytest 02-ddpm    # passes
```

## Commands

```
uv run modal run 02-ddpm/train.py      # train on Modal (profile ars-69317)
IMPL=model uv run pytest               # verify all reference implementations
```
