# ThomsonScatteringArbitrary

Rewrite of the `ThomsonScattering` package generalizing the forward model from
super-Gaussian electron/ion distributions to **arbitrary parametrized 1D
velocity distributions per species**, while keeping the whole pipeline
JAX-autodifferentiable (LBFGS/Adam fits, Tikhonov regularization, constraints,
SGLD posterior sampling, L-curve sweeps, CPU parallelism — all ported).

The original `ThomsonScattering/` package is untouched; the two coexist in
this repo. Decks without a `[species]` section run **identically** to the
original (every species defaults to `super_gaussian` through the same
tabulated dispersion function — verified bit-close, see Benchmarks).

## Quick start

```bash
# fits (same flags as the original thomson-fit CLI)
python -m ThomsonScatteringArbitrary.thomson_fit ThomsonScatteringArbitrary/examples/iaw_kappa/fit.toml

# forward model from [profiles]
python -m ThomsonScatteringArbitrary.thomson_forward ThomsonScatteringArbitrary/examples/forward_only/forward.toml
```

Choosing distributions in a deck:

```toml
[species]
electron = ["maxwellian"]                                   # registry model
ion      = [{ model = "kappa", x_max = 25.0, n_points = 2001 }]
# or a user JAX callable g(x, *shape_params), path relative to the deck:
# electron = ["my_dists.py:two_temp"]
```

Shape parameters become ordinary fit parameters named
`<param><e|i><species>` (`pe0`, `kappai0`, `fhote0`, …) and are configured via
`[params.*]` / `[penalty.*]` / `[constraints]` as usual. Full details and the
custom-callable contract: [DECK_API.md](DECK_API.md) (an addendum to the
repo-root `DECK_API.md`, which documents the unchanged base schema).

## How arbitrary distributions are handled

- **Analytic fast path** (`maxwellian`, `super_gaussian`): the original
  tabulated Z′(ζ, p) + incomplete-gamma feature term, numerically identical
  to the old package.
- **General path** (everything else): the user supplies the normalized 1D
  reduced distribution g(x) on x = (v−u)/√(2T/m) as a JAX scalar function.
  g′, g″ come from `jax.grad`; the susceptibility's principal-value integral
  is evaluated by singularity-subtracted composite-Simpson quadrature on a
  fixed grid with an exact log tail term (`dispersion.hilbert_disp`),
  matching the tabulated path to ~1e-5. Everything stays differentiable in
  both the moments (T, u, n, fractions) and the shape parameters.

## Layout

| file | role |
|---|---|
| `distributions.py` | `Distribution` abstraction, registry, custom-callable loader |
| `dispersion.py` | tabulated `_Zprime` (analytic path) + `hilbert_disp` quadrature (general path) |
| `forward.py` | spectral density / scattered power with a per-species model loop |
| `fitting.py` | parameter assembly (models' shape params included), transforms, optimizers |
| `deck.py` | TOML decks incl. `[species]`; HDF5 results |
| `sampling.py`, `l_curve.py`, `parallel.py`, `gain.py`, `plasma.py`, `arrays.py` | ports of the original modules |
| `thomson_fit.py`, `thomson_forward.py` | CLI entry points (`python -m …`) |
| `examples/` | original examples (unchanged decks) + `iaw_kappa`, `epw_custom_dist` |
| `benchmarks/` | parity + performance checks against the original package |

## Benchmarks (run from the repo root)

```bash
python -m ThomsonScatteringArbitrary.benchmarks.parity_dispersion
python -m ThomsonScatteringArbitrary.benchmarks.bench_maxwellian
python -m ThomsonScatteringArbitrary.benchmarks.smoke_sampling_lcurve
```

Results on the original example decks (`epw_basic`, `iaw_constraints`):
objective and gradient match the old package to machine precision; full fits
converge to identical losses in identical iteration counts with parameters
matching to ~1e-15; steady-state value-and-gradient evaluation is as fast or
faster than the original (the general quadrature path costs extra in
proportion to `Nt·Nk·n_points`).
