# tsadar vs. your `ThomsonScattering` package — comparison

A side-by-side reading of the two packages with concrete porting suggestions in both
directions.

- **tsadar** — `/home/bfoo/tsadar` (large, OMEGA-focused, JAX+Equinox+Optax)
- **yours**  — `/home/bfoo/.local/lib/python3.12/site-packages/ThomsonScattering/`
  (compact, ~900 lines, JAX+lmfit)

## 1. Birds-eye summary

| | tsadar | yours |
|---|---|---|
| Language/stack | JAX + Equinox + Optax + MLflow | JAX + lmfit |
| Lines of code | ~10k+ | ~900 |
| Forward model | Fully kinetic, `Z′` via numerical PV integral over actual f(v) | Fully kinetic, `Z′` via tabulated Re + analytic Im for super-Gaussian |
| EDF flexibility | Maxwellian → DLM → arbitrary 1D → arbitrary 2D → sph. harmonics (with NN) | Super-Gaussian only (electrons and ions), per-species `p` |
| Multi-species | electrons fixed Maxwellian; ions multi-species | Both multi-electron and multi-ion populations (each with own T, u, p, fract) |
| Time/streak handling | Batched shots via `vmap` | Time-profile parameters with **Tikhonov regularization in time** |
| Optimizer | Adam / L-BFGS / Optax with **autodiff** | lmfit Nelder-Mead by default (no AD even though forward model is JAX) |
| Loss | l1/l2/log-cosh/Poisson/covariance | mean-square (chi²) |
| Regularization | Moment penalties + monotonicity + hard smoothing of EDF | **Multi-order Tikhonov** with thresholds + monotonicity options |
| Uncertainty | Hessian via Equinox | (lmfit can compute, not invoked in this code) |
| Geometry | Scattering angle scalar | Full 3D `probe_vec`, `scatter_vec`, `ue_dir`, `ui_dir` |
| Aperture | Discrete weighted angle ensemble | Single angle |
| IRF | Real-space Gaussian convolve, peak-renorm | Per-time IRF array, vmap convolve |
| Background | 3-layer model added to forward model | Not modeled |
| Plasma gradients | linspace+mean over Tₑ, nₑ | Not present (but time-profile + Tikhonov is functionally similar across time) |
| Notch | Hard mask, ±3 nm around laser | Configurable wavelength range, `jnp.where`→ NaN |
| Streak warp | Bilinear-interp dewarp (EPW only) | Not present |
| Throughput | wavelength-dependent inverse-sensitivity | Not present (likely done externally) |
| Output | MLflow run artifacts | lmfit `MinimizerResult` |

## 2. Where each package is **stronger**

### What tsadar does that yours doesn't (yet)

1. **Distribution function flexibility.** Your code supports only super-Gaussian (with
   tunable `p`); tsadar can fit a fully arbitrary `f(v)` on a velocity grid, plus 2D
   anisotropic and spherical-harmonic expansions including a Mora–Yahi heat-flux closure.
   For DRESS / fast-ion or other non-Maxwellian work this matters.
2. **Autodiff-driven optimization.** tsadar wraps the loss in
   `equinox.filter_value_and_grad` and feeds analytic gradients to L-BFGS. Your code is
   JAX-based but uses lmfit's Nelder-Mead by default — gradient-free. For the scale you
   currently fit (handful of parameters) Nelder-Mead is OK, but if you ever expand to
   arbitrary EDFs it'll be unworkable.
3. **Multi-shot batching with `vmap`.** tsadar batches independent shots through one
   compiled forward model. Your time-profile pattern is similar in spirit, but if you
   ever want to fit multiple unrelated streaks in one run, batching would be cleaner.
4. **Hessian-based uncertainties.** tsadar inverts the loss Hessian for σ. lmfit can do
   covariance estimation if you switch to a least-squares method, but you'd need to call
   it.
5. **Background as a forward-model nuisance.** tsadar fits a parametric background per
   lineout and adds it to the model. If your data has any non-trivial background you're
   currently ignoring it (or pre-subtracting before calling `run_fit`).
6. **Finite aperture (collection f-number).** tsadar averages the spectrum over a
   discrete set of angles weighted by collection optics. You use a single
   `scatter_vec`, which assumes a point detector / pencil collection.
7. **IRF normalization.** tsadar peak-renormalizes after convolution so amplitude isn't
   coupled to PSF area. Your `Pklam = vmap(convolve)(Pklam, instr_func_arr)` doesn't —
   if your `instr_func_arr` isn't area-normalized, amplitude fits will be off.
8. **Streak warp + throughput.** tsadar handles these (incompletely, but they're there).
   If your data already has these baked in upstream, fine; if not, you're missing them.

### What your package does that tsadar doesn't

1. **Tikhonov regularization with threshold + relative scale + monotonicity.** This is
   strictly more sophisticated than what tsadar offers. Specifically:
   - Multi-order (0/1/2) penalties, configurable per parameter
   - Per-order **deadband threshold** so the penalty only kicks in past `threshold` —
     enables sharp features (shocks, ablation) without over-smoothing
   - **Relative scaling** (`relative_factor = 1 + relative*(|param|−1)`) so the penalty
     adapts to the local parameter magnitude
   - **Monotonic mode** (signed derivative) for cases where you know a quantity is
     monotonic in time
   - **Per-time, per-species, per-base** lookup hierarchy mirroring lmfit's parameter
     namespace
2. **Bayesian/MAP framing.** Your code explicitly composes
   `log_posterior = log_likelihood + log_prior` (with `log_prior` = the Tikhonov term).
   This is the right mental model and slots cleanly into a future MCMC upgrade. tsadar's
   penalties are not framed this way.
3. **L-curve scan.** `chi2_vary_tikhonov` runs the fit over a 2D grid of
   `(weight_scale, cutoff_scale)` and records chi² (likelihood-only) at each point. This
   is exactly the right tool for picking regularization strength, and tsadar has
   nothing equivalent.
4. **Multi-electron populations.** You support `Nelectrons > 1` (multiple electron
   species, each with own T, u, p, fract). tsadar treats electrons as a single species
   (with optional drift). For two-temperature electron plasmas this matters.
5. **General 3D geometry.** `probe_vec`, `scatter_vec`, `ue_dir`, `ui_dir` as 3-vectors
   with `jnp.dot` is more general than tsadar's scalar `θ_s`. Easier to use for arbitrary
   diagnostic placement and arbitrary drift directions.
6. **Cleaner forward-model surface.** Your `_spectral_density` is one well-typed function
   with a clear `[Nions, Nt, Nk]` broadcasting convention. tsadar's
   `FormFactor.__call__` is much harder to read because it threads through the parameter
   PyTree, gradient sampling, multi-angle weights, optional gain, etc.
7. **Far-zeta Laurent expansion of `Z′`.** Your `dispersion.py` switches between
   tabulated Re(Z′) and a Laurent series for `|ζ| > 10`, avoiding interpolation past
   the table edge. tsadar uses a rational-function approximation in
   [`ratintn.py`](tsadar/core/physics/ratintn.py); your approach is cleaner for the
   super-Gaussian-only case.
8. **Notch as `NaN`.** Using `jnp.where(mask, NaN, Pklam)` and `nanmean` in the loss is
   a clean way to mask without warping gradients. tsadar's `iawoff` zeros the spectrum,
   which biases the loss.

## 3. Concrete porting suggestions

### A. Things to port from tsadar → yours

In rough priority order:

**A1. Switch lmfit to a gradient-aware backend.** Your forward model is JAX, your
`_log_posterior` is differentiable, but you fit with Nelder-Mead. Two cheap upgrades:
- Use `lmfit`'s `least_squares` method (Levenberg–Marquardt) and provide a residual
  function instead of the scalar log-posterior. lmfit will compute Jacobians by finite
  difference — better than Nelder-Mead.
- Better: bypass lmfit entirely for the optimization and use `optax.lbfgs` or
  `scipy.optimize.minimize(method='L-BFGS-B', jac=True)` with `jax.grad` of your
  `_log_posterior`. Keep lmfit only for parameter bookkeeping. tsadar's pattern at
  [tsadar/inverse/loops.py](/home/bfoo/tsadar/tsadar/inverse/loops.py) is the template.

**A2. Add background as a fit parameter.** Add a per-time `bg_a`, `bg_b`, ... profile to
your `params` dict, evaluate a parametric background (constant, linear, quadratic, or
rat11), add it to `Pklam` inside `_compute_fit`, and let the regularization handle its
smoothness in time. Your existing Tikhonov framework will regularize the background
profile for free.

**A3. Finite aperture.** Replace `scatter_vec` with a (small) ensemble of vectors and
weights in `measurement_settings`. Inside `_spectral_density`, vmap over the angle axis
and dot-product with the weights at the end. Cost is O(angles) per evaluation. tsadar's
`sa_lookup` ([calibration.py:9](tsadar/utils/data_handling/calibration.py)) is the
template for the weights table.

**A4. IRF area renormalization.** After `vmap(convolve)(Pklam, instr_func_arr)`, divide
by the integral of `instr_func_arr` along the wavelength axis (or peak-renorm like
tsadar at [irf.py:115](tsadar/core/physics/irf.py)). Without it, amplitude fitting is
sensitive to how the IRF is normalized.

**A5. Hessian-based uncertainties.** Once A1 is done, you have a JAX scalar loss and a
parameter PyTree. `jax.hessian(loss)(params_arr)` → invert → diag → σ. Faster than
bootstrap, exact under Gaussian-likelihood + local-quadratic assumptions. tsadar's
[`get_sigmas`](tsadar/utils/process/postprocess.py#L153) is the template.

**A6. Plasma gradient sampling.** If you ever fit single-time data where probe-volume
gradients matter, tsadar's `linspace(1−g, 1+g, N) + mean` pattern is a one-line addition
inside `_spectral_density` and broadcasts naturally with your existing
`[Nions, Nt, Nk]` shape.

**A7. Arbitrary EDFs.** This is a much bigger lift, but if you need it: copy the
`Arbitrary1V` pattern from
[tsadar/core/modules/distribution_functions/base.py:180](tsadar/core/modules/distribution_functions/base.py#L180).
You'd also need to switch your `_Zprime` from analytic (super-Gaussian) to a numerical
PV integral over your tabulated `f(v)` — see
[`ratintn.py`](tsadar/core/physics/ratintn.py) and the imaginary-part `df/dv` evaluation
in [form_factor.py:259-289](tsadar/core/physics/form_factor.py#L259). Significant
work — only worth it if your physics demands it.

### B. Things to port from yours → tsadar (i.e. contributing back)

**B1. Tikhonov penalty plug-in.** This is the natural contribution. The cleanest
approach:

1. Add a config block under `optimizer.regularization`:
   ```yaml
   optimizer:
     regularization:
       fe:
         lambda_weights: [0.0, 1e-3, 1e-2]
         thresholds: [0, 0, 0]
         relative: true
         monotonic: [0, 0, 0]
       Te: { ... }   # if you also want temporal smoothness across batches
   ```
2. Add a `tikhonov_penalty` function next to the moment_loss block in
   [tsadar/inverse/loss_function.py:519](tsadar/inverse/loss_function.py#L519). Take
   `weights` PyTree and the regularization config; return a scalar.
3. Add it to `param_penalty` accumulation at `calc_loss`
   ([loss_function.py:317](tsadar/inverse/loss_function.py#L317)).

The function body is essentially what's in your `_tikhonov_penalty`, but operating on
JAX arrays inside the `weights` PyTree. The threshold/relative/monotonic options should
all carry over. Keep your nice **per-key fallback hierarchy** (specific → species →
base) since tsadar's parameter naming is similarly nested.

**B2. L-curve / chi² scan utility.** Port `chi2_vary_tikhonov`. tsadar can already do
multi-run via MLflow; the L-curve helper would be a small utility script that runs
`fit()` over a 2D `(weight_scale, threshold_scale)` grid and logs the resulting
data-only chi² to MLflow. The contribution is mostly the *idea* — that the right
regularization strength is the weakest one that doesn't significantly degrade chi².

**B3. Notch as `NaN`-mask, not zero.** tsadar's `iawoff` (in
[generate_spectra.py:232-240](tsadar/core/physics/generate_spectra.py#L232)) literally
zeros out a slice of the model spectrum, which then enters the loss as a residual of
`(data − 0)²`. That's a bias. Your `jnp.where(mask, NaN, Pklam)` + `jnp.nanmean(...)`
in the loss is correct. A small PR.

**B4. Multi-electron-population support.** Your `Nelectrons` extension is small and
well-encapsulated. If tsadar's user community has any two-temperature electron data this
is a real generalization. Bigger lift than the others because tsadar's parameter
classes are more entangled.

**B5. 3-vector geometry.** Replace tsadar's scalar `scattering_angle` with full
`probe_vec`, `scatter_vec`, drift vectors. Cleaner mental model and supports more
diagnostic geometries. Probably API-breaking though.

### C. Specific bugs / things to double-check in your package

A few things I noticed while reading:

**C1. Sign of econtr.** At [forward.py:124](forward.py#L124) you compute the electron
contribution as `|1 − sum_chiE / epsilon|^2`. The standard Sheffield form is
`|1 + sum_chiI / epsilon|^2 = |epsilon − sum_chiE|^2 / |epsilon|^2 = |1 − sum_chiE/epsilon|^2`,
so this is OK — but worth verifying against Sheffield 5.1.5 because the sign on the
ion contribution then flips through to your `icontr` line which uses
`|sum_chiE / epsilon|^2` (looks right too, but worth a check that your overall
spectrum matches a known case).

**C2. `Nelectrons` parameter unused.** Your `_spectral_density` signature includes a
trailing `Nelectrons=1` arg with a comment that it's unused. Consider removing — JAX
hashing on static args can hit this needlessly.

**C3. `_jitted_scattered_power_wavelength` uses static_argnames including `notch`.**
This means changing the notch range triggers a recompile. If you scan notches, mark it
non-static and use `jnp.where` outside the JIT'd part instead.

**C4. Variance in the likelihood.** `_log_likelihood = mean((fit-data)²/var)` — the
constant log(2π·var) term is dropped (fine for fitting), but if you ever compare
log-posteriors across different datasets or models, you'll need to keep it.

**C5. `ifract` semantics.** Comment at line 57 says "Note this is charge fraction not
ion number fraction" — easy to forget. Worth surfacing in the docstring.

**C6. `relative_factor = 1 + relative * (|param| − 1)`** at
[fitting.py:37](fitting.py#L37) — this is subtle. When `relative=False`, the factor is
1; when `relative=True`, it's `|param|`. So thresholds and norm_scale get multiplied by
the param magnitude. For `param < 1` (e.g. drift velocities near zero) this drives
thresholds toward zero, which is probably what you want, but for very small parameters
near zero crossings it might behave strangely.

## 4. Recommendation

Pick one direction and commit:

- **If you mostly want better fits in your own work:** prioritize A1 (autodiff
  optimizer), A2 (background), A4 (IRF norm), and A5 (Hessian σ). These are all small
  and high-value. A3 (aperture) only if your collection optics matter at the precision
  you need. Skip A7 unless you specifically need non-super-Gaussian EDFs.

- **If you want to contribute the regularization framework back to tsadar:** B1
  (Tikhonov penalty plug-in) is the cleanest contribution and would be genuinely
  novel — tsadar has nothing of comparable sophistication. B2 (L-curve helper) and
  B3 (NaN-notch) are nice quality-of-life follow-ups.

The `chi2_vary_tikhonov` + L-curve framing is probably your most valuable original
contribution. If I were the tsadar maintainer I'd take that PR in a heartbeat.
