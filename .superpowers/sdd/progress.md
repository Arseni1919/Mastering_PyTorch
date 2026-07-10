# 02-ddpm progress

base: d10924c

Task 1: complete (commits 5a84fc4..ba26ab6, review clean)
  - fixed in review: sys.modules registration in load_impl; pythonpath in pyproject
Task 2: complete (commits ba26ab6..25675b2, review clean)
  - Minor (deferred to final review): unused nn/F imports in model.py (consumed in Task 3)
  - Minor (deferred to final review): timestep_embedding drops a column on odd dim; unreachable, all call sites use dim=128
Task 3: complete (commits 25675b2..1979eb0, review clean)
  - attention verified vs naive einsum on non-square grid; permutation-equivariant
Task 4: complete (commits 1979eb0..0ff9d9c, review clean)
  - UNet 35,751,939 params; conditioning verified live at every ResBlock
  - user change: utils.py moved per-project (02-ddpm/utils.py); spec+CLAUDE.md updated
  - user change: UNet.__init__ split into _build_down/_build_up; bit-identical output verified
  - user request (pending): merge branch to main at the end
