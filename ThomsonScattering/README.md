# ThomsonScattering

Thomson-scattering forward model and fitting library with **arbitrary
parametrized 1D velocity distributions per electron/ion species**, fully
JAX-autodifferentiable: LBFGS/Adam MAP fits, Tikhonov regularization across
time, constraints with slack dummies, HMC/MALA/SGLD posterior sampling,
Laplace (MAP-Hessian) error bars, L-curve sweeps, and CPU parallelism
(time-axis sharding, chain-pmap, L-curve process pool).

This package replaced the original super-Gaussian-only implementation
(deleted at the same commit that renamed this one into its place; old↔new
parity was validated first — objective and gradient to machine precision,
full example fits converging to identical losses in identical iteration
counts). Decks without a `[species]` section run **identically** to the
original: every species defaults to `super_gaussian` through the same
tabulated dispersion function.

## Quick start

```bash
pip install -e .            # from the repo root; installs thomson-fit / thomson-forward

thomson-fit examples/iaw_kappa/fit.toml            # fit with a kappa ion species
thomson-fit examples/iaw_sample/fit.toml           # fit + posterior error bars
thomson-forward examples/forward_only/forward.toml # forward model from [profiles]
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
`[params.*]` / `[penalty.*]` / `[constraints]` as usual. Full schema and the
custom-callable contract: repo-root [DECK_API.md](../DECK_API.md).

## How arbitrary distributions are handled

- **Analytic fast path** (`maxwellian`, `super_gaussian`): tabulated Z′(ζ, p)
  + incomplete-gamma feature term.
- **General path** (everything else): the user supplies the normalized 1D
  reduced distribution g(x) on x = (v−u)/√(2T/m) as a JAX scalar function.
  g′, g″ come from `jax.grad`; the susceptibility's principal-value integral
  is evaluated by singularity-subtracted composite-Simpson quadrature on a
  fixed grid with an exact log tail term (`dispersion.hilbert_disp`),
  matching the tabulated path to ~1e-5. Everything stays differentiable in
  both the moments (T, u, n, fractions) and the shape parameters.

## Error bars

`[sampling]` in the deck (or `--sample`) estimates parameter uncertainties
after the MAP fit: `method = "laplace"` gives delta-method 1σ bars from the
MAP Hessian in seconds; `method = "mcmc"` (default) runs multi-chain MCMC —
`kernel = "hmc"` (default, trajectory-length-targeted with a Hessian mass
matrix), `"mala"`, or the legacy `"sgld"` (biased at finite step size; on the
`iaw_sample` example its error bars are ~2× wider than the agreeing
HMC/Laplace results). Degenerate (non-identified) parameter combinations are
detected from the MAP Hessian and reported with their physical loadings.

## Layout

| file | role |
|---|---|
| `distributions.py` | `Distribution` abstraction, registry, custom-callable loader |
| `dispersion.py` | tabulated `_Zprime` (analytic path) + `hilbert_disp` quadrature (general path) |
| `forward.py` | spectral density / scattered power with a per-species model loop |
| `fitting.py` | parameter assembly (models' shape params included), transforms, optimizers |
| `deck.py` | TOML decks incl. `[species]`; HDF5 results |
| `sampling.py` | HMC/MALA/SGLD kernels, Laplace covariance, diagnostics |
| `l_curve.py`, `parallel.py`, `gain.py`, `plasma.py`, `arrays.py` | regularization sweep, CPU parallelism, gain correction, plasma helpers |
| `benchmarks/` | dispersion parity, fit regression + timing, sampling smoke |

CLI entry points live at the repo root (`thomson_fit.py`,
`thomson_forward.py`, installed as `thomson-fit` / `thomson-forward`);
examples live in the repo-root `examples/`.

## Benchmarks (run from the repo root)

```bash
python -m ThomsonScattering.benchmarks.parity_dispersion
python -m ThomsonScattering.benchmarks.bench_maxwellian
python -m ThomsonScattering.benchmarks.smoke_sampling_lcurve
```
