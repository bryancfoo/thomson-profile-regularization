# `iaw_l_curve` — Tikhonov L-curve sweep on the D + C IAW fit

Same physics deck as [`../iaw_constraints/`](../iaw_constraints/) (multi-ion
fit with `[constraints]` + `[[extra_params]]`), with two additions:

1. `[penalty.Te0]` / `[penalty.Ti0]` — 1st- and 2nd-derivative Tikhonov
   terms on the time profiles of Te and Ti. The relative weighting between
   `L1` and `L2`, and between Te and Ti, is the *shape* of the regularizer.
2. `[l_curve]` — sweeps a scalar `lambda_scale` over the base
   `lambda_weights` (11 log-spaced points from `1e-2` to `1e+2`) and picks
   the corner of the residual-vs-penalty curve as the optimal trade-off.
   Replaces the single MAP fit; the optimal-λ result lands in `/best_fit`
   and `/params/*` as usual.

```bash
python examples/data/make_data_iaw.py    # one-time
thomson-fit fit.toml                     # ~2–5 min depending on hardware
python plot.py
```

## What gets written

- `fit_result.h5` — standard top-level `/best_fit`, `/params/*`, plus an
  `/l_curve` group with `lambda_scale`, `residual_norm`, `penalty_norm`,
  `curvature`, `loss`, `best_fits` (N×Nk×Nt), `params/<prefix>` (N×Nt),
  and the warm-start unregularized fit under `/l_curve/unreg/`. See
  [DECK_API.md §5](../../DECK_API.md) for the schema.
- `l_curve.png` — bare residual-vs-penalty plot with the corner marked
  (written directly by `thomson-fit` from `[l_curve].plot_path`).
- `spectra_optimal.png` + `params_vs_time_optimal.png` — written by
  `thomson-fit` via the `[plotting]` section, from the optimal-λ fit only.
- `params_vs_time.png` + `spectra.png` + `l_curve_profiles.png` — written
  by `plot.py`. The last figure is the interesting one: it overlays the
  Te₀ and Ti₀ profiles across all `lambda_scale` values (color-coded),
  emphasizing the optimal-λ trajectory in red and the synthetic truth in
  dotted black, so the under- / over-smoothing trade-off is immediately
  visible alongside the L-curve itself.

## Warm-starting

`[l_curve].warm_start = true` (the default) runs a single unregularized
fit first (all `lambda_weights = [0, 0, 0]`) and reuses its per-prefix
parameter profiles as the initial guess for every sweep point. This is far
more robust than starting each fit from the deck's literal initial guess
and avoids path-dependence between sweep points. The unregularized fit is
stored under `/l_curve/unreg/` for inspection.

## When to use this

If you have penalty terms in your deck and aren't sure how to set
`lambda_weights`, run this style of sweep on your problem: set the
*relative* weights between prefixes / derivative orders to reflect your
prior beliefs about smoothness, then let the L-curve pick the global
scale. The corner of the curve is the canonical "best compromise"
between data fidelity (low residual) and prior compliance (low penalty
norm). The per-λ profile overlay in `l_curve_profiles.png` is a good
sanity check that the corner pick actually matches your visual intuition.
