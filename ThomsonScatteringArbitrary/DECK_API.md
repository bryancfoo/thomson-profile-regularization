# ThomsonScatteringArbitrary — deck API addendum

This package is a rewrite of `ThomsonScattering` whose forward model accepts an
**arbitrary parametrized 1D velocity distribution per electron/ion species**,
while staying fully JAX-autodifferentiable. The deck schema is a **superset**
of the original: every section documented in the repo-root `DECK_API.md`
(`[data]`, `[measurement]`, `[probe_beam]`, `[params.*]`, `[[extra_params]]`,
`[constraints]`, `[penalty.*]`, `[fit]`, `[output]`, `[sampling]`,
`[l_curve]`, `[plotting]`) works unchanged. A deck with no `[species]` section
behaves identically to the original package (every species defaults to
`super_gaussian`, with its `pe`/`pi` exponent parameters).

CLI invocation (run from the repo root):

```bash
python -m ThomsonScatteringArbitrary.thomson_fit     path/to/fit.toml
python -m ThomsonScatteringArbitrary.thomson_forward path/to/forward.toml
```

All flags from the original `thomson-fit` (`--sample`, `--l-curve`,
`--n-workers`, `--n-devices`, `--serial`) work the same way.

---

## The `[species]` section

Selects a distribution model per species. Lists must match `Nelectrons` and
`len(ion_z)`.

```toml
[species]
electron = ["maxwellian"]
ion      = ["super_gaussian", { model = "kappa", x_max = 25.0, n_points = 2001 }]
# or a custom callable (path relative to the deck file):
# electron = ["my_dists.py:two_temp"]
```

Each entry is one of:

| form | meaning |
|---|---|
| `"maxwellian"` | Maxwellian, no shape params (analytic fast path) |
| `"super_gaussian"` | super-Gaussian of order `p` ∈ [2, 5] (analytic fast path — tabulated Z′, identical to the original package; shape param `p`) |
| `"kappa"` | kappa / Lorentzian-tailed distribution (general path; shape param `kappa`, κ > 3/2) |
| `"super_gaussian_numeric"` | the super-Gaussian routed through the general quadrature — for validation only |
| `"file.py:function"` | custom callable (general path; see contract below) |
| inline table `{ model = "...", x_max = ..., n_points = ... }` | any of the above plus quadrature options |

**Quadrature options** (general-path models only): `x_max` (half-width of the
normalized-velocity grid; default 10, kappa default 20 — raise it for
fat-tailed distributions) and `n_points` (Simpson points; default 2001, forced
odd). Accuracy of the dispersion integral scales as `(2·x_max/n_points)²`;
runtime and memory scale linearly in `n_points`.

### Shape parameters become fit parameters

Each model's shape parameters join the fit under the prefix
`<name><e|i><species_idx>`:

- super-Gaussian `p` on electron species 0 → `pe0` (same name as the original package)
- `kappa` on ion species 1 → `kappai1`
- custom `two_temp(x, fhot, rhot)` on electron species 0 → `fhote0`, `rhote0`

They are configured through `[params.*]`, `[penalty.*]`, `[constraints]`, and
appear in the output HDF5 `/params/<prefix>` exactly like the moment
parameters. The three-level specificity (`fhote0_3` > `fhote0` > `fhote`)
applies as usual. Defaults come from the model (registry defaults, or the
callable's Python default arguments); parameters without a model default must
be given a `value` in the deck.

```toml
[params.kappai0]
value = 5.0
min   = 1.7
max   = 30.0
vary  = true
```

The original package's check that super-Gaussian exponents satisfy `min >= 2`
is still enforced, but only for species actually using the analytic
`super_gaussian` model (a custom model with a parameter named `p` sets its own
range).

### Custom-callable contract

```python
# my_dists.py
import jax.numpy as jnp

def two_temp(x, fhot=0.1, rhot=3.0):
    """g(x): normalized 1D reduced distribution, ∫ g dx = 1."""
    cold = (1.0 - fhot) * jnp.exp(-x**2) / jnp.sqrt(jnp.pi)
    hot  = fhot * jnp.exp(-x**2 / rhot) / jnp.sqrt(jnp.pi * rhot)
    return cold + hot
```

- **First argument**: the normalized parallel velocity `x = (v − u)/vth`,
  `vth = sqrt(2·T/m)`, where `T` and `u` are the species' temperature and
  drift fit parameters. The function receives scalars (the package `vmap`s
  it), so no broadcasting logic is needed.
- **Remaining arguments**: shape parameters. Names must not contain
  underscores; defaults become fit-parameter defaults.
- **Return**: the 1D distribution *reduced along the scattering wavevector k*
  (already projected — this is what the scattering physics needs and is the
  most general convention: anisotropic/beam distributions are expressible).
  Normalize so `∫ g dx = 1`; a Maxwellian is `exp(−x²)/√π` (variance ½). If
  your family's `T` convention differs (e.g. kappa), document it — `vth` is
  whatever `sqrt(2·T/m)` gives for the fitted `T`.
- Must be **JAX-differentiable** in `x` (always; the dispersion integral uses
  `jax.grad` for g′ and g″) and in any shape parameter left free in the fit.
  Avoid non-smooth constructs (`jnp.where` branches that produce `0·inf` in
  gradients, `abs` powers below 2 at the origin, …).

For isotropic 3D distributions f(|v|), supply the 1D reduction
`g(x) = 2π ∫_|x| f(s) s ds` yourself (the built-in `super_gaussian` /
`maxwellian` already are the correct isotropic reductions).

### How it enters the physics

For each species, with ζ = (ω − k·u)/(k·vth):

```
chi_s   = wp_s²/(vth_s·k)² · [ P∫ g′(x)/(ζ − x) dx + iπ·g′(ζ) ]
S(k,ω) ∝ Σ_s  2π/(k·vth_s) · |screening_s|² · g_s(ζ_s)
```

Analytic models use the original tabulated Z′(ζ,p) (bit-identical results);
general models evaluate the principal-value integral by singularity-subtracted
Simpson quadrature on the fixed grid, with an exact log tail term. Validated
against the table to ~1e-5 relative (`benchmarks/parity_dispersion.py`).

---

## Forward decks (`thomson_forward`)

`[profiles]` works as before; shape-parameter profiles are looked up as
`<name><e|i><idx>` first, then `<name><e|i>` (scalar, `(Nt,)`, or
`(Nspecies, Nt)`) — so existing `pe` / `pi` keys keep working. A `[species]`
section selects models exactly as in fit decks.

**Unit fix:** the original `thomson_forward.py` passed `ne` (cm⁻³) and
temperatures (eV) to the forward model without converting to SI — a latent
unit bug. This package's forward CLI converts (`×1e6`, `×e/kB`), matching
`thomson-fit` and the deck documentation. Forward decks tuned against the old
CLI's raw pass-through will produce different (now-correct) output.

---

## Behavioral notes & gotchas

- **Pin or bound everything you don't fit.** Free parameters with unbounded
  defaults (`ue`, `ui`, `efract`, `ifract` left `vary = true` with no
  `min`/`max`) produce NaN gradients — same behavior as the original package.
  The example decks pin them explicitly.
- **Sampling preconditioner:** when any species uses the general path, the
  `diag_hessian` / `full_hessian` preconditioners skip `jax.hessian` (its
  compile through the quadrature is prohibitive) and use the
  finite-difference fallback automatically.
- **Performance:** the analytic path matches the original package
  (machine-identical spectra; per-eval as fast or faster). The general path
  costs roughly `Nt·Nk·n_points` array work per objective evaluation — e.g.
  ~50–130 ms per value-and-gradient at `Nt=8, Nk=200, n_points=1001–4001` on
  CPU, scaling linearly in each factor. Use the smallest `n_points` your
  accuracy needs, and the existing `--n-devices` time-sharding for long
  streaks.
- **L-curve / multiprocessing:** model specs are stored as plain
  strings/dicts in `measurement_settings` and resolved inside each worker
  process, so `[l_curve]` with `n_workers > 1` works with registry models and
  custom callables alike (custom paths are resolved to absolute at deck load).

---

## Provided examples (`ThomsonScatteringArbitrary/examples/`)

| deck | demonstrates |
|---|---|
| `epw_basic`, `iaw_constraints`, `iaw_sample`, `iaw_l_curve`, `iaw_full`, `forward_only` | the original examples, unchanged decks (default super-Gaussian models), reading the shared data in `examples/data/` |
| `iaw_kappa` | `[species]` with the built-in `kappa` ion model; fits `Te0`, `Ti0`, `kappai0` against `data/make_data_kappa.py` synthetic truth |
| `epw_custom_dist` | a user-supplied callable (`my_dists.py:two_temp`, bi-Maxwellian electrons); fits the hot-electron fraction `fhote0` |

Benchmarks (`ThomsonScatteringArbitrary/benchmarks/`):
`parity_dispersion.py` (quadrature vs table, forward parity vs the
original package), `bench_maxwellian.py` (fit-level parity + timing on the
original example decks), `smoke_sampling_lcurve.py` (sampler + L-curve on the
general path).
