# LeWorldModel

## TODOs

- to understand this:
```
define_model.py:48 — added TranslationPredictor (z_{t+1} = z_t + emb(a), zero-init)
- Your MLP predictor can memorize an arbitrary permutation of the 225 states, so loss_pred → 0 while latent distance stays meaningless (probe R² 0.63, 57% success — and lowering λ further made it worse, 25.5%).
- Restricting actions to pure translation means the only way to predict every transition is for the latent to be affine in (row, col).
- Real LeWM doesn't need this — pixels and a ViT supply smoothness that a one-hot toy can't.
```
- to make run inference script to run multi-step prediction and not only a single step forward
- to understand `run_probe.py` script
- to understand SigReg function 
- what happens exactly at the point where the nice converged latent space goes crazy without SigReg
- to experiment with SigReg lam value
- to experiment with different seeds that can squeeze the initial convergence
- to experiment with different sizes and shapes of the map
- to experiment with different reg functions
- to experiment with different predictors
- to experiment with more complex domains closer to real applications
- to achieve 100% success rate in latent space - look closely at the cases where it misses the target location
- to experiment with continual learning concept where the agent for example needs to learn fast a new skill


## DONEs
- to understand scaling logic of lum
```
Answer:
Tune by the probe, not the loss. Print both terms every step, start at lam ≈ loss_pred / sigreg, then sweep ×3 in each direction and pick by probe R². That's what actually separated 0.43 from 0.99 here — the loss curves alone looked fine in both cases, since loss_pred reached ~0.002 in the failing runs too.
```

