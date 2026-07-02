# Speed notes: thomson-profile-regularization vs tsadar

Reference notes from a 2026-05-22 investigation into why a 71×651 EPW fit on
`FoilGasDist25B/116349_early` is projected at ~13 h on CPU vs. "minutes" for
tsadar on similar problems. The `_unpack` vectorization (item 4 below) was
implemented; the others are deferred until profiling confirms they're worth it.

Both packages use JAX + optax + autodiff, so the gap is not a tech-stack
mismatch — it's a handful of specific architectural and configuration
differences. Ranked roughly by expected speedup ÷ effort.

## 1. Hardware: CPU vs GPU

tsadar is GPU-first (multi-GPU sharding via `NamedSharding` for 2D angular
fits). The current 13 h estimate is CPU-only.

The forward model in `ThomsonScattering/forward.py` is already fully
vectorized JAX, so a single GPU should give **10–50× with no code change**.
This is almost certainly the largest single lever.

To try: install `jax[cuda12]` (or whatever matches the CUDA on the box) and
confirm with `jax.devices()` — JAX will move arrays automatically.

## 2. Problem size per forward call

This package fits **all 651 time slices jointly** in one optimization
(~1953 free params for the 116349 deck), because the Tikhonov regularization
in `ThomsonScattering/fitting.py` (`_tikhonov_penalty`, ~L47–84) couples them
across time.

Per forward call: 10 angles × 1 electron × 651 t × 71 λ ≈ 462k spectral
density evals, plus IRF convolution at `forward.py:363` (`vmap` of 71×651
conv ops).

tsadar's typical "few minutes" config (`tsadar/configs/arts-2d/defaults.yaml`)
uses `batch_size: 6` lineouts with `lineout.skip: 50`, so it fits ~4 lineouts
× 1000 epochs = 4k forward passes on a 6-lineout problem — roughly **100×
less work per forward call** than the joint streak fit here.

The "few minutes" tsadar quote almost certainly refers to that batched
workflow, not a joint streak fit. The joint fit is a real feature of this
package (regularization across time), not a bug.

If runtime is still bad after items 1 + 3 + 4 below: **window the streak**
into overlapping chunks of e.g. 64 slices with Tikhonov stitching at
boundaries. Each chunk's LBFGS has ~192 params instead of ~1953, so per-iter
cost drops ~10× and LBFGS converges faster in lower dimension. Adds
non-trivial code and risks discontinuities at chunk boundaries — only worth
it if the cheaper fixes don't get you to acceptable runtime.

## 3. Optimizer cost per iteration

The 116349 deck uses `sgld_lbfgs` with `max_iter=5000` (`sgld_iter=1000` + up
to 4000 LBFGS). At ~1953 free parameters:

- LBFGS + optax Zoom line search (`fitting.py:541`) re-evaluates the forward
  model 1–5× per iteration → 4k–20k forward calls.
- tsadar uses Adam with cosine LR for a fixed 1000 epochs → exactly 1000
  forward calls.

Even before the per-call cost difference, this is potentially 4–20× more
forward calls.

To try:
- `sgld_iter=200, max_iter=1500` and see if final loss is comparable.
- Cap LBFGS line search: `optax.scale_by_zoom_linesearch(max_linesearch_steps=8)`.
- Plain Adam with cosine LR after SGLD warmup.

## 4. `_unpack` trace size (FIXED 2026-05-22)

`_unpack` and `_build_params_dict` in `fitting.py` used to do
`jnp.stack([_get(x, f"{prefix}_{t}") for t in range(Nt)])` for each of ~9
parameter prefixes. With `Nt=651` that's ~9×651 indexing + stack ops in the
JAX trace per call. JIT compile time scales with trace size, and the
`sgld_lbfgs` path JITs twice (once for `_sgld_step`, once for the LBFGS
`step`).

Replaced with precomputed per-prefix gather tables:

```python
gather_tables[prefix] = (idx_arr, fixed_arr, is_free)  # all (Nt,)
def _gather_prefix(x, prefix):
    idx_arr, fixed_arr, is_free = gather_tables[prefix]
    return jnp.where(is_free, x[idx_arr], fixed_arr)
```

Constrained prefixes keep using their callable. Numerical result is
unchanged — same gather, same `jnp.where`, just one op per prefix instead
of `Nt`.

## 5. Compute-bound primitives (probably fine, not investigated deeply)

- `_Zprime` (`dispersion.py:27–78`) — tabulated 2D interpax in the small-|ζ|
  branch and a Laurent series in the large branch. Comparable to tsadar's
  tabulated Z′.
- `gammaincc` (`forward.py:140, 153`) — JAX's implementation is OK; runs in
  every forward call.
- IRF conv via per-t `vmap(jnp.convolve)` (`forward.py:363`) — Nk=71 is small
  enough that FFT-based conv wouldn't help.

## 6. K-smearing angles

The 116349 deck uses 10 scattering angles with what looks like Gauss-style
quadrature weights. Test 5 angles via Gauss-Legendre on the same θ range
([FoilGasDist25B/116349_early/make_k_smear.py](FoilGasDist25B/116349_early/make_k_smear.py)
is the generator) — if profiles don't change meaningfully, that's a free 2×.

## Suggested order of attack

1. **Run on GPU** — install `jax[cuda12]`, no code change.
2. **Profile** with `jax.profiler.trace()` to break down compile vs
   steady-state vs SGLD vs LBFGS. Before doing anything else, find out where
   the time actually goes.
3. **Cut `sgld_iter`/`max_iter`, cap LBFGS line search.**
4. ~~Vectorize `_unpack`~~ — done.
5. **Test 5 k-smearing angles** instead of 10.
6. **Temporal windowing** of the streak — largest code change, last resort.

## Files referenced

- `ThomsonScattering/forward.py` (k-smearing `vmap` at L295–337, IRF conv at
  L363, `_spectral_density` at L39–168)
- `ThomsonScattering/fitting.py` (`_build_grad_problem`, `_unpack`,
  `_build_optimizer`, `run_fit_grad`, `_run_sgld_phase`)
- `FoilGasDist25B/116349_early/116349_epw.toml` (deck for the slow fit)
- `tsadar/tsadar/core/physics/form_factor.py`,
  `tsadar/tsadar/inverse/loss_function.py`,
  `tsadar/tsadar/inverse/loops.py`,
  `tsadar/configs/arts-2d/defaults.yaml`

---

# Implemented: CPU parallelism (2026-05-31)

Hardware reality check on the dev box (measured): the only GPU is an NVIDIA
**NVS 315** (Fermi, compute 2.1) — a display-only card modern JAX/CUDA cannot
use, and no CUDA toolkit is installed. So item 1 above ("run on GPU") is **not
possible on this machine**; it needs a different box. The real resource here is
**48 CPU threads (24 cores), no cgroup quota**. A single forward+grad uses only
~3–6 of those 48 cores (XLA-CPU under-utilizes this op-mix; even an ideal matmul
tops out ~13), so the levers are (A) run independent fits in parallel processes
and (B) shard one fit across CPU "devices". Both keep numerics identical.

## A. L-curve sweep → parallel processes  (`ThomsonScattering/parallel.py`)
The sweep points in `compute_L_curve` are independent (all warm-start from one
unregularized fit), so they run across a spawn-based `ProcessPoolExecutor`, each
worker pinned to a disjoint CPU-core block.
- Control: `--n-workers N` (CLI) / `[l_curve].n_workers` (deck) /
  `compute_L_curve(..., n_workers=)`. **Default 1 (sequential) — parallelism is
  opt-in.** `n_workers` is the number of concurrent worker *processes* (each fit
  uses ~3–4 cores), NOT a core count; `0`/negative auto-sizes to `cores // 4`.
- **Bit-identical** to the sequential sweep (process isolation; verified `Δ=0`,
  including a 2-species deck with `[constraints]` + `[[extra_params]]` — the
  string constraints and extra-param arrays pickle across the process boundary
  and reproduce exactly).
- Measured (6 points, single-angle 1024×74, max_iter=50): **1.75×** wall
  (1336 s → 764 s), cores 2.6 → 5.1. Lower than the structural `(N+1)/2` ceiling
  because this forward is **memory-bandwidth bound** — 6 concurrent fits starve
  each other (~1 core each vs 2.6 solo) — plus the serial warm-start (~25% of the
  parallel wall) and compile-heavy short fits. Expect better scaling on the real
  10-angle data (≈10× more compute per memory load) and with realistic max_iter
  (compile amortized).

## B. Single fit → time-axis sharding  (`fitting.py: _make_sharded_nll`)
The forward model is independent per time slice, so the data-fidelity
forward+grad is sharded over the Nt axis with `shard_map`; only the masked chi²
sum + valid-pixel count are `psum`-reduced. The Tikhonov penalty stays unsharded
on the full (small) param arrays. Nt is padded to a multiple of the device count
with NaN data columns (masked out → contribute 0).
- **Per-time measurement arrays must be sharded too.** `instr_func_arr` (Nk, Nt)
  is padded+sharded along time and substituted into a per-shard copy of
  `measurement_settings` — otherwise the forward's per-time IRF conv vmap sees a
  full-Nt IRF against an Nt-block spectrum (the bug first hit on real EPW data).
  `bg` (background, K+1×Nt) is likewise sharded. `throughput`/`wavelengths` are
  per-*wavelength* and broadcast fine. CAVEAT: a per-time `normalization_scale`
  (Nt,) would hit the same issue — keep it scalar (the default). Validated
  fp-identical (rel ~1e-15) at 1 vs 4 devices on a deck *with* an IRF + padding.
- Control: `--n-devices N` (CLI) / `THOMSON_CPU_DEVICES=N` (env). **Off by
  default** (`use_time_shard = jax.device_count() > 1`). Must be set before the
  first `import jax` (handled in `__init__.py` / `thomson_fit._bootstrap_cpu_devices`).
- **fp-identical** (differs only by float64 summation order: rel ≈ 1e-16 on loss,
  1e-15 on grad — the absolute diff looks large only because the integral-norm
  loss is ~3e16).
- Measured (10-angle 691×72 forward+grad, 8 devices): **2.28×** (4677 → 2070 ms),
  cores 5.6 → 13.4. Uses the box better than the process pool (small per-device
  time-block → better cache locality).

## Kill-switch & guards
- `--serial` / `THOMSON_NO_PARALLEL=1` forces everything serial (no pool, no
  sharding) regardless of other flags/deck — single predicate `serial_requested()`.
- **Do not combine** `--n-devices > 1` with `--n-workers > 1` (sweep workers would
  inherit the device count and oversubscribe). They're alternatives: sharding for
  one big streak fit, the pool for sweeps.
- The SGLD sampler always builds its objective with `shard_time=False`: the
  per-chain `vmap` does not compose with the objective's `shard_map`. Instead,
  chains are parallelized directly.

## C. Sampling → chain-pmap  (`sampling.py: _make_pmapped_step`)
SGLD iterations are inherently sequential (Markov), so the only parallel axis is
the chains. With >1 device and ≥1 device per chain (and parallelism not killed),
`run_sgld_posterior` maps the per-chain step across devices with `pmap` (one
chain per device) instead of `vmap`-ing on one device. The per-chain keys are
still generated host-side, so it is numerically identical per step (device
placement changes neither the math nor the RNG; verified per-step `rel ~5e-16`).
- Auto-selected when `device_count > 1 and n_chains <= device_count and not
  serial`; falls back to the single-device `vmap` otherwise. Enabled by the same
  `--n-devices N` that drives time-sharding (for `--sample`, `--n-devices` speeds
  up both the MAP fit via time-sharding and the chains via pmap).
- Measured: **1.83×** (4 chains across 4 devices, 200.9 s → 109.8 s); per-step
  fp-identical (`rel ~5e-16`), posterior means statistically identical. Modest
  like the L-curve pool — the per-chain forward is memory-bandwidth bound, so the
  4 concurrent chains contend. NOTE: individual sample *trajectories* diverge over
  many steps (SGLD is stochastic and amplifies the per-step ~1e-16 fp difference),
  so the posterior is statistically identical, not bitwise.

## GPU readiness (deferred)
Code stays device-agnostic JAX. On a real-GPU box: `pip install jax[cuda12]`,
confirm `jax.devices()`. Single-GPU → `device_count == 1` → sharding stays off and
the forward just runs on the GPU. The `float(val)` host-sync each LBFGS iter
(`fitting.py` `_run_loop`) is fine on CPU; on GPU, consider checking convergence
every k iters.

## Test assets (not in this repo; under /tmp during development)
- `/tmp/ts_test/epw_test.toml`, `iaw_test.toml` — HTPD-synthetic decks with an
  `[l_curve]` section (data copied from `~/HTPD/data`).
- Spike/validation scripts: `spike_shard.py` (time-shard equivalence + speedup),
  `validate_lcurve.py` / `bench_lcurve.py` (pool equivalence + speedup),
  `validate_shard.py` (wired 1-vs-N-device equivalence).

---

# Implemented: general-path forward + sampler speedups (2026-07-02)

## Quadrature (general-distribution) forward model
Measured on the kappa example (Nt=8, Nk=200, Nx=1001, 1 angle), val_and_grad:
**54.0 → 28.9 ms (1.87×)**, objective bit-identical, gradient rel ~4e-14
(float reassociation from batching). Three changes in
`ThomsonScattering/distributions.py` + `forward.py`:

1. **Fused `disp_and_reduced`** — the forward model needs both the dispersion
   integral and g(zeta) per species; they now share one traversal of the
   time axis instead of two.
2. **`time_batch` control of the time-axis map** (`_map_time`). Default
   `"auto"`: vectorize the whole time axis (`lax.map` in one chunk ≈ vmap)
   when the estimated quadrature slab fits `AUTO_MEM_BUDGET_BYTES` (4 GiB
   incl. fudge for AD residuals/aperture batching), else fall back to the
   sequential row map. Measured (Nt=64 tiled kappa): sequential 317 ms,
   full-vmap 247 ms, but intermediate chunks (8/16/32) 420–431 ms — this
   box is memory-bandwidth bound and chunked scan+vmap is the worst of both
   worlds, hence auto picks only the two extremes. Override per species:
   `i_models = [{model = "kappa", time_batch = 16}]`.
3. **Constant g′-grid precompute** for distributions with no shape
   parameters (custom fixed-form g): evaluated once at model build instead
   of per time row per call.

## Sampler (`ThomsonScattering/sampling.py`)
- **Chunked `lax.scan` main loops** for all kernels: the old loop paid one
  jitted call + a host sync per iteration (`float(r_t)`, `np.asarray(us)`);
  now the only host↔device traffic is one transfer per ~100-iteration chunk,
  with step-size adaptation on-device. The pmap chain backend runs the same
  scan inside `pmap` (cross-chain reductions via `pmean`/`all_gather`).
- **Batched FD Hessian setup**: the 2·D finite-difference gradient
  evaluations (needed on the general path where `jax.hessian` is
  compile-prohibitive) run through `lax.map(batch_size=16)` instead of a
  serial Python loop; the Hessian is computed once and reused for the
  preconditioner and the Laplace covariance.
- **HMC / MALA kernels** (`kernel = "hmc"` default): fewer, less-correlated
  samples for the same error bars, no step-size bias, and acceptance-rate
  dual averaging replaces the drift/noise heuristic. `method = "laplace"`
  skips MCMC entirely (MAP Hessian + delta method) — error bars in seconds.
