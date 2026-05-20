# ThomsonScattering — deck API & user guide

A compact JAX + optax package for fitting time-resolved Thomson-scattering
streaks with regularization in time. The forward model supports super-Gaussian
EDFs for an arbitrary number of electron and ion species, finite collection
aperture, instrument response, throughput, background, and the Turnbull
SRS/SBS gain correction. Fits run via the gradient-based `run_fit_grad`
optimizer (LBFGS / Adam / AdamW / SGLD→LBFGS).

This document covers the TOML "deck" format consumed by the two CLI entry
points and the small public Python API.

---

## 1. Install & CLI

```bash
pip install -e .
```

Two entry points are registered:

```
thomson-fit       path/to/fit_deck.toml      # run a fit from a deck
thomson-forward   path/to/forward_deck.toml  # compute and plot a streak
```

Both also accept being called via `python thomson_fit.py ...` /
`python thomson_forward.py ...`, and both fall back to an interactive
prompt when no argument is provided.

Three example decks live under [`example/`](example/):

| Deck | Demonstrates |
|---|---|
| [`example/fit_deck.toml`](example/fit_deck.toml) | minimal antiStokes-EPW fit, single ion, LBFGS |
| [`example/fit_deck_constraints.toml`](example/fit_deck_constraints.toml) | multi-ion IAW + `[constraints]` + `[[extra_params]]` + SGLD→LBFGS |
| [`example/fit_deck_full.toml`](example/fit_deck_full.toml) | "kitchen sink": IRF + throughput + `notch` + `background_order` + `[probe_beam]` + `[penalty.*]` + Adam |

Run them in order:

```bash
python example/make_data.py        # generates example/data.h5 (EPW)
python example/make_data_iaw.py    # generates example/data_iaw.h5 (IAW, used by decks 2 & 3)
thomson-fit example/fit_deck.toml
thomson-fit example/fit_deck_constraints.toml
thomson-fit example/fit_deck_full.toml
```

---

## 2. Conventions

| Quantity | Unit |
|---|---|
| Wavelength (`probe_wavelength`, `wavelengths`, `notch`) | **meters** |
| Density (`n`) | cm⁻³ |
| Temperature (`Te`, `Ti`) | eV |
| Velocity (`ue`, `ui`) | m/s |
| Geometry vectors (`probe_vec`, `scatter_vec`, `ue_dir`, `ui_dir`) | unit vectors |
| `probe_diameter` (in `[probe_beam]`) | µm |
| `probe_intensity` (in `[probe_beam]`) | W/cm² |

Time is whatever unit the `profile_axis` / `time` arrays use — the
penalty machinery just sees `np.diff(profile_axis)`. Use ns to match the
example decks.

**Array file references.** Any deck field that expects an array can be
either an inline list, or a path:

```toml
wavelengths = [261.0, 261.5, 262.0, ...]        # inline
wavelengths = "data.h5:wavelengths"             # HDF5 dataset
wavelengths = "lambda.npy"                      # numpy binary
wavelengths = "throughput.csv"                  # whitespace-separated (or comma)
```

Paths are resolved relative to the deck file.

**Parameter naming.** All free parameters carry a `_<time-step>` suffix.
Per-species parameters also carry a `<species-index>` suffix between the
base name and the time index. Examples:

```
n_3              # total density at t = 3        (no species index)
Te0_5            # electron-species 0, time t = 5
Ti2_0            # ion-species 2, time t = 0
ifract1_7        # ion-species 1 charge fraction at t = 7
bg0_3            # background poly coefficient 0 at t = 3
```

When you specify a `[params.X]` table the key uses three levels of
specificity (most → least specific): `"Te0_3"` > `"Te0"` > `"Te"`. The
most-specific match wins.

---

## 3. Fit deck schema

The same parser (`build_settings_from_deck` in
[`ThomsonScattering/deck.py`](ThomsonScattering/deck.py)) reads every fit
deck. Sections:

### `[data]` (required)

```toml
[data]
path        = "data.h5"
pkl_dataset = "Pkl_data"          # (Nk, Nt)
var_dataset = "Pkl_var"           # (Nk, Nt)
time_axis   = "data.h5:time"      # optional — required if any [params.X]
                                  # uses source_time_axis interpolation.
```

### `[measurement]` (required)

```toml
[measurement]
Nelectrons         = 1                          # # of electron species
ion_z              = [1, 6]                     # atomic numbers
ion_a              = [2, 12]                    # atomic mass (amu)
probe_wavelength   = 2.6325e-7                  # meters
probe_vec          = [0.0, 0.0, 1.0]            # unit vec, probe k̂
scatter_vec        = [0.8660254, 0.0, 0.5]      # unit vec, scatter k̂
ue_dir             = [1.0, 0.0, 0.0]            # electron drift dir
ui_dir             = [1.0, 0.0, 0.0]            # ion drift dir
wavelengths        = "data.h5:wavelengths"      # (Nk,) meters

# Optional:
instr_func_arr     = "data.h5:irf"              # (Nk,) or (Nk, Nt) IRF
irf_normalization  = "area"                     # "area" | "peak" | "none"
throughput         = "throughput.csv"           # (Nk,) sensitivity
aperture_weights   = [0.25, 0.5, 0.25]          # (Nangles,) — needs (Nangles, 3) scatter_vec
notch              = [263.249e-9, 263.251e-9]   # mask wavelengths inside
background_order   = 1                          # adds bg0..bgK params
normalization_type = "max"                      # "max" | "sum" | "integral"
normalization_scale = 1                         # multiplier on the norm
```

### `[probe_beam]` (optional, disables when absent)

Probe-beam SRS/SBS amplification correction
(Turnbull et al., PRL 136, 135101 (2026)).

```toml
[probe_beam]
intensity_W_per_cm2 = 9.2e14
diameter_um         = 165.0
pol_p_fraction      = 1.0          # 1.0 = p-pol, 0.0 = s-pol, 0..1 mix
gain_mode           = "exact"      # "exact" | "small_gain" | "off"
```

`gain_mode = "off"` leaves the parser running (so the deck still validates)
but skips the multiplicative correction.

### `[params.<prefix>]` — initial guess, bounds, and fix/free

```toml
[params.Te0]
value = 500.0       # initial guess
min   = 10.0        # lower bound (default: -inf)
max   = 5000.0      # upper bound (default: +inf)
vary  = true        # free / fixed flag
```

**Array warm starts.** Supply an array reference for `value` to set
distinct initial guesses per time step:

```toml
[params.Te0]
value = "prev_result.h5:Te0"     # (Nt,) array, one value per time step
min   = 10.0
max   = 5000.0
vary  = true
```

The array must have shape `(Nt,)` matching the data time dimension.

**Cross-time-axis warm starts.** When the warm-start array lives on a
different time grid (e.g. an EPW fit feeding into an IAW fit), add
`source_time_axis`; the parser linearly interpolates onto the data's
time axis.

```toml
[data]
time_axis = "iaw_data.h5:time"   # (Nt,) target axis — required

[params.Te0]
value            = "epw_result.h5:Te0"     # (Nt_src,) on source axis
source_time_axis = "epw_result.h5:time"    # (Nt_src,)
min = 10.0
max = 5000.0
vary = true
```

**Relative bounds.** Set bounds that scale with each time step's value:

```toml
[params.Te0]
value   = "epw_result.h5:Te0"
rel_min = -0.10                  # min = value * 0.90 per time step
rel_max = +0.10                  # max = value * 1.10 per time step
vary    = true
```

### `[[extra_params]]` — free dummy variables referenced from `[constraints]`

```toml
[[extra_params]]
name  = "ifract1_floor"
value = 0.10
min   = 0.0
max   = 0.5
vary  = true
```

Each entry is replicated across all time steps as `ifract1_floor_0`,
`ifract1_floor_1`, ... and accessible from constraint expressions by its
bare prefix (`ifract1_floor`).

### `[constraints]` — derived parameters

Keys are parameter prefixes; values are string expressions written in
terms of other prefixes (no `_<t>` suffix — substitution happens
automatically per time step).

```toml
[constraints]
Ti1     = "Ti0"                                  # equality coupling
ifract1 = "max(ifract1_floor, 1 - ifract0)"      # sum-to-one with floor
ifract5 = "1 - ifract0 - ifract1 - ifract2 - ifract3 - ifract4"
```

Functions available inside expressions: `min`, `max`, `abs`, `where`,
`clip`, `sqrt`, `exp`, `log`, plus arithmetic. Constrained prefixes are
removed from the free-variable vector — their values are derived at
forward-eval time from the constrained-prefix expression.

### `[penalty.<prefix>]` — Tikhonov regularization

Penalizes 0th, 1st, and 2nd derivatives of a parameter profile in time.

```toml
[penalty.Te0]
profile_axis   = "data.h5:time"   # (Nt,) — sets the derivative grid
lambda_weights = [0.0, 1.0, 0.5]  # [L0, L1, L2]
thresholds     = [0.0, 0.0, 0.0]  # only penalize |deriv| > threshold
relative       = true             # rescale by parameter magnitude
monotonic      = 0                # 0 = symmetric, +1 = penalize decreases,
                                  #               -1 = penalize increases
norm_scale     = 1                # or [s0, s1, s2] for per-order scaling
```

If a global key (`[penalty.Te]`) is provided, it applies to every
species (`Te0`, `Te1`, ...). Per-species keys override the global. `n`
has no species index — use `[penalty.n]`.

### `[fit]` — optimizer & convergence

```toml
[fit]
optimizer = "lbfgs"          # "lbfgs" | "adam" | "adamw" | "sgld_lbfgs"
max_iter  = 1000             # hard iteration cap
tol       = 1e-8             # window-based convergence (see below)
lr        = 1e-2             # learning rate for adam / adamw

# sgld_lbfgs-only:
sgld_iter        = 300       # SGLD-phase iterations (default: max_iter // 2)
sgld_lr          = 1e-3      # SGLD step size
sgld_noise_scale = 0.1       # initial noise σ added to the gradient
sgld_noise_decay = 0.55      # noise decay rate (0 = constant)
sgld_seed        = 0         # RNG seed for reproducibility
```

**Convergence rule.** The loop tracks a rolling window of the last 100
loss values. When the relative improvement (`(window[0] - min(window)) /
|window[0]|`) drops below `tol`, the fit terminates with `success=True`.
Hitting `max_iter` first returns `success=False`.

Any keys not listed above are forwarded as kwargs to the optax
constructor (e.g. `memory_size` for LBFGS, `weight_decay` for AdamW).

### `[output]`

```toml
[output]
path = "fit_result.h5"
```

Default (if omitted) is `<deck_stem>_result.h5` next to the deck file.

### `[sampling]` — posterior sampling (optional)

Runs preconditioned SGLD after the MAP fit, producing posterior summary
statistics and (optionally) a sidecar HDF5 of raw samples. Triggered by
`enabled = true` here or by the `--sample` CLI flag on `thomson-fit`.

```toml
[sampling]
enabled         = true
n_samples       = 1000        # post-burn-in samples per chain
n_chains        = 4
burn_in         = 1000        # iterations before sampling; default = n_samples
thin            = 1
temperature     = "auto"      # "auto" | "unit" | positive float
step_size       = 0.1         # initial; adapted in burn-in if adapt_step
adapt_step      = true
adapt_target    = 0.3         # median drift/noise ratio to target
precond         = "diag_hessian"   # "diag_hessian" | "full_hessian" | "rmsprop" | "identity"
perturb_scale   = 1.0         # per-chain init offset in posterior-std units
seed            = 0
polish_map      = false       # only useful if temperature ≈ 1
save_samples    = true
samples_path    = "auto"      # "auto" → <output stem>_samples.h5
save_cross_corr = true        # write the full (P·Nt × P·Nt) corr matrix
```

**Temperature semantics.** `"auto"` resolves to `2 / N_pixels_valid`, which
rescales the user's per-pixel-mean `V_fit = mean(r²/σ²) + Σ λ_o·mean(d_o²)`
into the proper Gaussian negative log-likelihood `0.5·sum(r²/σ²)` plus an
implicit prior `Σ (N_pix·λ_o/2)·mean(d_o²)`. The MAP location is preserved
(uniform scaling). Set `temperature = 1.0` ("unit") to sample from
`exp(-V_fit + log|J|)` directly — uncertainty intervals then depend on the
loss convention; rebinning changes their width.

**Outputs.** Posterior summary stats land under `/summary/` in the primary
HDF5 (means, stds, 16/50/84 percentiles, intra-prefix correlations, R-hat,
ESS, optional full cross-correlation matrix). Full samples land in the
sidecar file at `samples_path` (default `<output stem>_samples.h5`). All
samples are constraint-resolved, so `samples/ifract1` is the physical
quantity `max(floor, 1 - ifract0)`, not the raw dummy.

### `[plotting]` (CLI extension, consumed by `thomson_fit.py`)

```toml
[plotting]
shot_num       = 12345
init_png       = "init.png"
streak_png     = "streak.png"
profiles_png   = "profiles.png"
profile_layout = "epw"           # "epw" | "iaw"
```

All four PNG outputs are optional and skipped when not specified.

### CLI deck extensions (consumed before `build_settings_from_deck`)

These two extensions in `thomson_fit.py` resolve into the standard
`[measurement]` keys before parsing:

```toml
[measurement.throughput_xlsx]
path = "throughput.xlsx"
lam_col       = "Lambda"
value_col     = "Sensitivity No Grating"
lam_unit      = "nm"             # "nm" | "m"
gaussian_sigma = 0               # optional smoothing

[measurement.irf_hdf4]
path           = "irf.hdf"
dataset        = "Streak_array"
center_index   = 512
gaussian_sigma_2d = [1.0, 1.0]   # optional 2D smoothing
flip_wavelength = false
slice_mode     = "uniform"       # "uniform" | "per_slice"
N_avg          = 50              # required for "per_slice"
```

---

## 4. Forward deck schema

A separate, smaller deck format consumed by
[`thomson_forward.py`](thomson_forward.py). Sections are identical to
the fit deck where they overlap, except for `[profiles]` (which replaces
`[data]` + `[params]`):

```toml
[profiles]
time   = [...]                   # (Nt,) — also sets the time axis
ne     = [...]                   # (Nt,) cm⁻³
Te     = [...]                   # (Nelectrons, Nt) or (Nt,) eV
Ti     = [...]                   # (Nions, Nt) eV
ue     = [...]                   # m/s
ui     = [...]                   # (Nions, Nt) m/s
pe     = [...]                   # super-Gaussian exponent
pi     = [...]                   # (Nions, Nt) super-Gaussian exponent
efract = [...]
ifract = [...]                   # (Nions, Nt) — should sum to 1 over ions

[measurement]
# same as fit-deck [measurement]

[probe_beam]
# optional, same as fit-deck

[output]
path      = "streak.png"
data_path = "forward.h5"         # optional — also save Pklam/time/wavelengths

[plotting]
figsize = [12, 6]
dpi     = 150
cmap    = "viridis"
```

---

## 5. Output HDF5 layout

`save_fit_results` writes:

```
/best_fit                  (Nk, Nt) forward model at best-fit parameters
/time                      (Nt,)    optional — time axis if supplied
/params/n                  (Nt,)    one dataset per parameter prefix
/params/Te0                (Nt,)
/params/Ti0                (Nt,)
/params/Ti1                (Nt,)    constrained / fixed prefixes also written
/params/ifract0            (Nt,)
/params/ifract1_floor      (Nt,)    extra-params written too
/params/bg0                (Nt,)    background coefs if any
...
```

When posterior sampling ran, an additional `/summary` group is written:

```
/summary/mean/<prefix>                  (Nt,)        posterior mean
/summary/std/<prefix>                   (Nt,)        posterior std
/summary/p16/<prefix>                   (Nt,)
/summary/p50/<prefix>                   (Nt,)
/summary/p84/<prefix>                   (Nt,)
/summary/correlations/<prefix>          (Nt, Nt)     intra-prefix Pearson r
/summary/rhat/<prefix>                  (Nt,)
/summary/ess/<prefix>                   (Nt,)
/summary/cross_correlations             (P·Nt, P·Nt) full matrix (optional)
/summary/cross_correlations_labels      (P·Nt,) str  row/col labels
attrs: n_chains, n_samples, burn_in, thin, temperature, step_size_final,
       precond, max_rhat, min_ess, wall_time_s, ...
```

A sidecar `<output stem>_samples.h5` (path configurable in `[sampling]`)
holds the raw chains:

```
/samples/<prefix>          (n_chains, n_samples, Nt) constraint-resolved
/u_samples                 (n_chains, n_samples, D)  raw u-space samples
/log_probs                 (n_chains, n_samples)
/step_size_history         (burn_in,)
/varying_keys              (D,) string
/u_chain_init              (D,)
```

File-level attributes:

```
loss        : final scalar loss
nit         : number of iterations
success     : True if window-based tolerance was reached before max_iter
deck_toml   : verbatim copy of the deck (for provenance)
```

---

## 6. Python API

For programmatic use, the public surface re-exported from
[`ThomsonScattering/__init__.py`](ThomsonScattering/__init__.py):

```python
from ThomsonScattering import (
    load_deck,                  # parse a TOML deck → dict
    build_settings_from_deck,   # dict → (data, var, meas, pen, pars, fit, ...)
    run_fit_grad,               # the fit driver
    save_fit_results,           # HDF5 writer
    compute_initial_fit,        # forward model at the initial guess
    build_params,               # build a {name: Param} dict from settings
    Param,                      # dataclass: value, min, max, vary
    scattered_power_wavelength, # forward model (units: see § 2)
    spectral_density,           # S(k, omega) (no instrument response)
)
```

Typical scripted use:

```python
deck = load_deck("my_deck.toml")
(Pkl_data, Pkl_var, meas, pen, pars, fit_kw,
 extras, constraints, out_path) = build_settings_from_deck(deck)

result, best_fit = run_fit_grad(
    Pkl_data, Pkl_var, meas,
    penalty_settings=pen,
    params_settings=pars,
    fit_settings=fit_kw,
    extra_params=extras,
    constraints=constraints,
    progress=True,
)

save_fit_results(out_path, result, best_fit, time_axis=meas.get("time"))
```

`result` is a `SimpleNamespace` with:

- `result.params_dict` — `{prefix: (Nt,) np.ndarray}` including all
  constrained, extra, and fixed parameters
- `result.fun` — final loss
- `result.nit`, `result.success`
- `result.x`, `result.varying_keys` — flat vector of free parameters in
  physical space, plus their names

---

## 7. Provided examples — quick reference

[`example/make_data.py`](example/make_data.py) generates
[`example/data.h5`](example/data.h5): a clean single-ion antiStokes-EPW
streak (200 wavelengths × 10 time steps), no IRF or throughput baked in.

[`example/make_data_iaw.py`](example/make_data_iaw.py) generates
[`example/data_iaw.h5`](example/data_iaw.h5): a D + C IAW streak
(200 wavelengths × 10 time steps) with a Gaussian IRF (FWHM ≈ 25 pm) and
a wavelength-dependent throughput envelope baked in. The same throughput
is also written to [`example/throughput.csv`](example/throughput.csv).

The three fit decks use:

- **deck 1**, [`example/fit_deck.toml`](example/fit_deck.toml): `data.h5`,
  no IRF/throughput. Free params: `n`, `Te0`. Optimizer: LBFGS.
- **deck 2**, [`example/fit_deck_constraints.toml`](example/fit_deck_constraints.toml):
  `data_iaw.h5`, applies `instr_func_arr` and `throughput` (both loaded
  from the HDF5 file). Free params: `Te0`, `Ti0`, `ifract0`,
  `ifract1_floor`. `[constraints]` sets `Ti1 = Ti0` and `ifract1 =
  max(ifract1_floor, 1 - ifract0)`. Optimizer: SGLD → LBFGS.
- **deck 3**, [`example/fit_deck_full.toml`](example/fit_deck_full.toml):
  `data_iaw.h5`, applies IRF (from HDF5), throughput (from CSV — same
  numbers as the HDF5 version, demonstrating the CSV file-loading path),
  `notch`, `background_order = 1`, `[probe_beam]` (with `gain_mode =
  "off"` so the section is parsed but the correction is disabled), and
  `[penalty.*]` Tikhonov terms. Optimizer: Adam.

The single forward deck
[`example/forward_deck.toml`](example/forward_deck.toml) pairs with
`data.h5`: recomputes the synthetic EPW streak directly from the
profiles, useful for sanity-checking that geometry / wavelength window
agree with the fitter's view.
