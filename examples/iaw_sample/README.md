# `iaw_sample` — IAW with constraints + posterior sampling

Same physics deck as [`../iaw_constraints/`](../iaw_constraints/) but with a
`[sampling]` section that triggers preconditioned SGLD after the LBFGS MAP.

```bash
python examples/data/make_data_iaw.py   # one-time
thomson-fit fit.toml                    # writes fit_result.h5 (+ fit_result_samples.h5)
python plot.py                          # error bands rendered from posterior
```

The same `plot.py` from the no-sampling example is used; it auto-detects
the `/summary/` group in `fit_result.h5` and renders 16/84-percentile bands
on each parameter trajectory. Constraint-resolved samples mean the
`ifract1` band reflects the *physical* quantity
`max(ifract1_floor, 1 - ifract0)`, not the raw dummy.

The full chains land in `fit_result_samples.h5` (a few MB) for downstream
analysis — see [DECK_API.md §5](../../DECK_API.md) for the schema.
