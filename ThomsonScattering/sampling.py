"""Preconditioned SGLD posterior sampler for Thomson-scattering fits.

Builds on :func:`ThomsonScattering.fitting._build_grad_problem`: takes the
``SimpleNamespace`` it returns and produces a JAX-jitted, Jacobian-corrected
sampling target plus a multi-chain SGLD runner with R-hat/ESS diagnostics
and per-sample constraint resolution.

Public API
----------
- :func:`build_sampling_problem` — wrap a fit problem with a sampling target.
- :func:`run_sgld_posterior` — top-level multi-chain sampler.

Notes on the sampling target
----------------------------
The fit objective is ``V_fit(x) = mean(r^2/σ^2) + Σ λ_o · mean(d_o^2)``.
The sampler targets

    log π(u) = -V_fit(T(u)) / T_temp + log|det dT/du|

where ``T_temp`` defaults to ``2 / N_pixels_valid``. At that temperature
``-V_fit/T_temp`` is the proper Gaussian negative-log-likelihood
``0.5 · sum(r^2/σ^2)``, and the penalty becomes an implicit Gaussian prior
``Σ (N_pix · λ_o / 2) · mean(d_o^2)``. Uniform scaling of V_fit preserves
the LBFGS MAP location; the only mode shift comes from the Jacobian term,
and at the default temperature it is ~10⁻⁵ in u-space (well below a
posterior std).

Rolling our own SGLD (vs. blackjax) keeps full control over the bijector
Jacobian and adds no new dependency.
"""
from __future__ import annotations

import time
from functools import partial
from types import SimpleNamespace

import jax as _jax
_jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, value_and_grad, vmap, grad

from .fitting import _log_det_jac_u


# ─── sampling-problem wrapper ───────────────────────────────────────────────

def _resolve_temperature(temperature, problem):
    """Resolve ``temperature`` from the deck/CLI into a positive float.

    Accepts: ``None`` or ``"auto"`` (default 2/N_pixels_valid),
    ``"unit"`` (legacy T=1), or any positive number.
    """
    if temperature is None or temperature == "auto":
        mask = jnp.isnan(problem.Pkl_data) | jnp.isnan(problem.Pkl_var)
        n_pix = float(jnp.sum(~mask))
        if n_pix <= 0:
            raise ValueError("No valid (non-NaN) data pixels found for temperature auto-init.")
        return 2.0 / n_pix
    if temperature == "unit":
        return 1.0
    t = float(temperature)
    if not (t > 0):
        raise ValueError(f"temperature must be positive, got {temperature!r}")
    return t


def build_sampling_problem(problem, *, temperature=None):
    """Wrap a fit problem with a Jacobian-corrected, temperature-rescaled
    sampling target.

    Parameters
    ----------
    problem : SimpleNamespace
        Output of :func:`ThomsonScattering.fitting._build_grad_problem`.
    temperature : float, ``"auto"``, ``"unit"``, or None
        Sampling temperature. ``"auto"`` (default) → ``2 / N_pixels_valid``,
        which makes ``-V_fit/T`` the proper Gaussian negative log-likelihood
        and gives standard physics-style 1σ error bars.

    Returns
    -------
    SimpleNamespace with the original ``problem`` plus:
        target_log_prob(u) -> scalar
        target_log_prob_and_grad(u) -> (logp, glogp)
        log_det_jac(u) -> scalar
        resolve_one(u) -> dict[prefix -> (Nt,) jnp.ndarray]
        temperature : float
        n_pixels_valid : int
    """
    T = _resolve_temperature(temperature, problem)
    lower = problem.lower
    upper = problem.upper

    mask = jnp.isnan(problem.Pkl_data) | jnp.isnan(problem.Pkl_var)
    n_pix = int(jnp.sum(~mask))

    def log_det_jac(u):
        return _log_det_jac_u(u, lower, upper)

    def target_log_prob(u):
        return -problem.objective_u(u) / T + log_det_jac(u)

    log_det_jac_jit = jit(log_det_jac)
    target_log_prob_jit = jit(target_log_prob)
    target_log_prob_and_grad_jit = jit(value_and_grad(target_log_prob))

    @jit
    def resolve_one(u):
        x_phys = problem.to_external_jax(u)
        return problem.build_params_dict(x_phys)

    return SimpleNamespace(
        problem=problem,
        temperature=T,
        n_pixels_valid=n_pix,
        log_det_jac=log_det_jac_jit,
        target_log_prob=target_log_prob_jit,
        target_log_prob_and_grad=target_log_prob_and_grad_jit,
        resolve_one=resolve_one,
    )


# ─── preconditioner ─────────────────────────────────────────────────────────

def _build_diag_hessian_precond(problem_s, u_ref, *, floor=1e-6):
    """Diagonal preconditioner from |H(target_log_prob)(u_ref)|.

    Returns ``M_diag`` of shape (D,) with ``M_diag[i] = 1 / max(|H_ii|, floor)``.
    """
    H = _jax.hessian(problem_s.target_log_prob)(u_ref)
    h_diag = jnp.abs(jnp.diag(H))
    return 1.0 / jnp.maximum(h_diag, floor)


def _build_full_hessian_precond(problem_s, u_ref, *, reg=1e-6):
    """Full Hessian preconditioner ``M = (|H| + reg·I)^{-1}``.

    Builds an SPD approximation by taking the absolute eigenvalues, so the
    preconditioner is well-defined even where the Hessian has saddle-point
    directions. Returns ``(M_matrix, L_chol)`` for SGLD's noise step
    (``L_chol L_chol^T = M``).
    """
    H = _jax.hessian(problem_s.target_log_prob)(u_ref)
    # symmetric eigendecomposition; |λ| gives SPD ≈ -H for sampling-target convention
    w, V = jnp.linalg.eigh(0.5 * (H + H.T))
    w_abs = jnp.maximum(jnp.abs(w), 0.0) + reg
    M = (V * (1.0 / w_abs)) @ V.T
    L = (V * (1.0 / jnp.sqrt(w_abs))) @ V.T
    return M, L


# ─── SGLD step kernels ──────────────────────────────────────────────────────

def _sgld_step_diag(u, key, eps, M_diag, target_grad_fn):
    """One preconditioned-SGLD step with diagonal M.

    u' = u + 0.5*eps*M*g + sqrt(eps*M) * N(0, I),  g = grad(log p)(u)

    Returns (u_new, logp, g) so the caller can use g for step-size adaptation
    without paying for a second gradient evaluation.
    """
    logp, g = target_grad_fn(u)
    noise = jr.normal(key, u.shape, dtype=u.dtype)
    u_new = u + 0.5 * eps * M_diag * g + jnp.sqrt(eps * M_diag) * noise
    return u_new, logp, g


def _sgld_step_full(u, key, eps, M_full, L_chol, target_grad_fn):
    """One preconditioned-SGLD step with full M (M = L L^T)."""
    logp, g = target_grad_fn(u)
    noise = jr.normal(key, u.shape, dtype=u.dtype)
    u_new = u + 0.5 * eps * (M_full @ g) + jnp.sqrt(eps) * (L_chol @ noise)
    return u_new, logp, g


# ─── multi-chain runner (vmapped) ───────────────────────────────────────────

def _make_vmapped_step(target_grad_fn, kind):
    """Return a jit+vmap step function for the given preconditioner kind."""
    if kind == "diag":
        def step(us, keys, eps, M_diag):
            return vmap(_sgld_step_diag, in_axes=(0, 0, None, None, None))(
                us, keys, eps, M_diag, target_grad_fn
            )
    elif kind == "full":
        def step(us, keys, eps, M_full, L_chol):
            return vmap(_sgld_step_full, in_axes=(0, 0, None, None, None, None))(
                us, keys, eps, M_full, L_chol, target_grad_fn
            )
    else:
        raise ValueError(f"Unknown step kind: {kind!r}")
    return jit(step)


def _drift_noise_ratio(g_arr, eps, M_diag_or_full, kind):
    """Median over (chains × coords) of |drift_i| / |noise_i| as adapt proxy.

    g_arr : (n_chains, D)

    Median (not mean) keeps the adapt robust to a few stiff coordinates with
    huge gradients — without it, one outlier dominates the mean and the
    adapt shrinks eps to a value that's safe for the outlier but too small
    for everyone else. The full-Hessian preconditioner addresses the same
    problem geometrically, but the median heuristic helps the diagonal case.
    """
    if kind == "diag":
        drift = 0.5 * eps * jnp.abs(M_diag_or_full[None, :] * g_arr)
        noise = jnp.broadcast_to(
            jnp.sqrt(eps * M_diag_or_full)[None, :], drift.shape
        )
    else:
        drift = 0.5 * eps * jnp.abs(g_arr @ M_diag_or_full.T)
        # Use sqrt(eps * diag(M)) as per-coord noise std.
        diag_M = jnp.diag(M_diag_or_full)
        noise = jnp.broadcast_to(
            jnp.sqrt(eps * diag_M)[None, :], drift.shape
        )
    ratio = drift / (noise + 1e-30)
    return jnp.median(ratio)


# ─── diagnostics ────────────────────────────────────────────────────────────

def _rhat(samples):
    """Rank-normalized Gelman-Rubin R-hat.

    samples : array of shape (n_chains, n_samples, *)
    Returns array of shape (*) — R-hat per coordinate. Coordinates whose
    samples are effectively constant (relative std < 1e-10) return NaN.
    """
    x = np.asarray(samples)
    nc, ns = x.shape[0], x.shape[1]
    if nc < 2 or ns < 2:
        return np.full(x.shape[2:], np.nan)
    chain_means = x.mean(axis=1)                                    # (nc, *)
    grand_mean = chain_means.mean(axis=0)                            # (*)
    B = ns * np.var(chain_means, axis=0, ddof=1)
    W = np.mean(np.var(x, axis=1, ddof=1), axis=0)
    var_hat = ((ns - 1) / ns) * W + B / ns
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / np.maximum(W, 1e-300))
    # Mask coordinates whose samples are effectively constant.
    overall_std = x.reshape(-1, *x.shape[2:]).std(axis=0)
    scale = np.maximum(np.abs(grand_mean), 1.0)
    rhat = np.where(overall_std > 1e-10 * scale, rhat, np.nan)
    return rhat


def _ess(samples):
    """Effective sample size (per coordinate).

    samples : (n_chains, n_samples, *)
    Truncates the autocorrelation sum at the first non-positive pair-sum
    (simplified Geyer initial monotone sequence). Constant-variance coords
    (e.g. fixed parameters) return NaN so they don't poison min-ESS reports.
    """
    x = np.asarray(samples)
    nc, ns = x.shape[0], x.shape[1]
    if nc < 1 or ns < 4:
        return np.full(x.shape[2:], np.nan)

    mu = x.mean(axis=1, keepdims=True)
    y = x - mu                                                       # (nc, ns, *)
    var_c = (y * y).mean(axis=1)                                     # (nc, *)
    var_avg = var_c.mean(axis=0)                                     # (*)

    # Effectively-constant coords: float64 jitter on a fixed value gives
    # tiny but nonzero variance; the FFT autocorrelations are pure noise
    # and produce spurious ESS values. Mask those out.
    overall_mean = x.reshape(-1, *x.shape[2:]).mean(axis=0)
    scale = np.maximum(np.abs(overall_mean), 1.0)
    constant_mask = ~(np.sqrt(var_avg) > 1e-10 * scale)

    nfft = 1 << int(np.ceil(np.log2(2 * ns)))
    Fy = np.fft.rfft(y, n=nfft, axis=1)
    acf = np.fft.irfft(Fy * np.conj(Fy), n=nfft, axis=1)[:, :ns] / ns
    acf_mean = acf.mean(axis=0)                                       # (ns, *)
    safe_var = np.where(var_avg > 0, var_avg, 1.0)
    rho = acf_mean / safe_var
    rho[0] = 1.0

    n_pairs = (ns - 1) // 2
    if n_pairs < 1:
        return np.full(x.shape[2:], float(nc * ns))
    pair = rho[1:1 + 2 * n_pairs].reshape(n_pairs, 2, *rho.shape[1:]).sum(axis=1)
    keep = (pair > 0)                                                 # (n_pairs, *)
    cum_keep = np.cumprod(keep, axis=0)
    pair_eff = pair * cum_keep
    tau = 1.0 + 2.0 * pair_eff.sum(axis=0)
    tau = np.maximum(tau, 1.0)
    ess = nc * ns / tau
    ess = np.where(constant_mask, np.nan, ess)
    return ess


# ─── top-level driver ──────────────────────────────────────────────────────

def run_sgld_posterior(problem, u_map, *,
                       temperature=None,
                       n_samples=1000, n_chains=4,
                       burn_in=None, thin=1, perturb_scale=1.0,
                       step_size=0.1, adapt_step=True, adapt_target=0.3,
                       precond="diag_hessian",
                       seed=0, progress=False,
                       polish_map=False, polish_max_iter=200):
    """Run multi-chain preconditioned SGLD.

    Parameters
    ----------
    problem : SimpleNamespace
        Output of :func:`ThomsonScattering.fitting._build_grad_problem`.
    u_map : array (D,)
        LBFGS MAP in unconstrained space (e.g. from running ``run_fit_grad``
        and re-encoding ``result.x`` via ``problem.to_internal_np``).
    temperature : float, ``"auto"``, ``"unit"``, or None
    n_samples : int
        Number of post-burn-in, post-thin samples per chain.
    n_chains : int
        Independent chains; init at u_map + perturb_scale * N(0, I).
    burn_in : int or None
        Burn-in iterations per chain. Default = n_samples.
    thin : int
        Keep every ``thin``-th sample after burn-in.
    step_size : float
        Initial SGLD step size in u-space. Adapted during burn-in if
        ``adapt_step`` is True.
    adapt_step : bool
        Robbins-Monro adaptation of step_size during burn-in only.
    adapt_target : float
        Target ratio of drift to noise during burn-in. 0.5–1 is the useful
        range; default 0.7 balances bias (small) vs. mixing (large).
    precond : {"diag_hessian", "full_hessian", "rmsprop", "identity"}
        Mass-matrix preconditioner.
        - ``diag_hessian`` (recommended default): inverse |diag(H)| at the
          init point. Captures per-coordinate curvature. Cheap (one
          Hessian-diagonal evaluation).
        - ``full_hessian``: inverse |H| via eigendecomposition. Captures
          cross-parameter correlations from Tikhonov regularization. Good
          for problems with strong inter-time correlations; O(D^3) once.
        - ``rmsprop``: running EMA of grad^2 across burn-in, frozen at the
          end. No Hessian needed but mixes poorly when parameter scales
          differ widely (the EMA on raw u-space gradients can't compensate
          for the bijector scale). Useful as a Hessian-free fallback.
        - ``identity``: no preconditioning. Only useful for diagnostics or
          when the problem is already nicely scaled.
    polish_map : bool
        If True, run a brief LBFGS on the Jacobian-corrected target to
        recenter chains on the posterior mode. Only meaningful when the
        Jacobian shift is non-negligible (i.e. ``temperature ≈ 1``).

    Returns
    -------
    SimpleNamespace (see module docstring for exhaustive field list).
    """
    if burn_in is None:
        burn_in = n_samples

    problem_s = build_sampling_problem(problem, temperature=temperature)
    u_map = jnp.asarray(u_map, dtype=jnp.float64)
    D = int(u_map.shape[0])

    # Optionally polish to the Jacobian-corrected mode.
    if polish_map:
        u_chain_init = _polish_map(problem_s, u_map, max_iter=polish_max_iter)
    else:
        u_chain_init = u_map

    # Preconditioner setup at u_chain_init.
    # ``rmsprop`` is adaptive: M_diag is updated during burn-in from an EMA
    # of (mean across chains of) grad^2, then frozen for sampling so the
    # sampler targets a fixed invariant distribution.
    kind = "diag" if precond in ("diag_hessian", "identity", "rmsprop") else "full"
    rmsprop_state = None
    if precond == "diag_hessian":
        M_diag = _build_diag_hessian_precond(problem_s, u_chain_init)
        precond_obj = M_diag
    elif precond == "identity":
        M_diag = jnp.ones(D, dtype=jnp.float64)
        precond_obj = M_diag
    elif precond == "rmsprop":
        M_diag = jnp.ones(D, dtype=jnp.float64)
        rmsprop_state = jnp.ones(D, dtype=jnp.float64)  # v_EMA initial
        precond_obj = M_diag
    elif precond == "full_hessian":
        M_full, L_chol = _build_full_hessian_precond(problem_s, u_chain_init)
        precond_obj = (M_full, L_chol)
    else:
        raise ValueError(f"Unknown precond: {precond!r}. Choose "
                         "'diag_hessian', 'full_hessian', 'rmsprop', or 'identity'.")

    # Build the vmapped step function.
    step_fn = _make_vmapped_step(problem_s.target_log_prob_and_grad, kind)

    # Initialize chain states: perturb in sigma_u units (sqrt of diag precond)
    # so perturb_scale=1.0 means "start chains roughly 1 posterior std apart
    # from the MAP". This is appropriate for R-hat sensitivity.
    key = jr.PRNGKey(seed)
    key, sk = jr.split(key)
    if kind == "diag":
        sigma_u = jnp.sqrt(precond_obj)
    else:
        # full Hessian: use sqrt(diag(M_full)) as per-coord std
        sigma_u = jnp.sqrt(jnp.diag(precond_obj[0]))
    perturb = perturb_scale * sigma_u[None, :] * jr.normal(
        sk, (n_chains, D), dtype=jnp.float64
    )
    us = u_chain_init[None, :] + perturb                              # (n_chains, D)

    eps = float(step_size)
    eps_history = np.zeros(burn_in, dtype=np.float64)

    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(desc="SGLD burn-in", total=burn_in + n_samples * thin,
                   unit="iter")

    # ─── burn-in (with optional adaptation) ──────────────────────────────
    t0 = time.time()
    rmsprop_beta = 0.95
    rmsprop_eps = 1e-6
    for it in range(burn_in):
        key, *sub = jr.split(key, n_chains + 1)
        chain_keys = jnp.stack(sub)
        if kind == "diag":
            us, logps, gs = step_fn(us, chain_keys, eps, precond_obj)
            M_for_ratio = precond_obj
        else:
            us, logps, gs = step_fn(us, chain_keys, eps, precond_obj[0], precond_obj[1])
            M_for_ratio = precond_obj[0]

        if rmsprop_state is not None:
            # Update RMSProp EMA from this step's gradients, freshen M_diag.
            # Mean across chains keeps the preconditioner shared (and the
            # sampler well-defined). M_diag freezes when burn-in ends.
            g_sq = jnp.mean(gs ** 2, axis=0)
            rmsprop_state = rmsprop_beta * rmsprop_state + (1 - rmsprop_beta) * g_sq
            precond_obj = 1.0 / (jnp.sqrt(rmsprop_state) + rmsprop_eps)

        if adapt_step:
            r_t = _drift_noise_ratio(gs, eps, M_for_ratio, kind)
            r_t_f = float(r_t)
            # Robbins-Monro: r = drift/noise grows with sqrt(eps), so to push
            # r toward target we move eps in the direction of (target - r).
            # η decays through burn-in so eps settles.
            eta = 0.05 / (1.0 + 10.0 * (it / max(1, burn_in)))
            eps = eps * float(jnp.exp(eta * (adapt_target - r_t_f) / max(adapt_target, 1e-3)))
            eps = float(np.clip(eps, 1e-12, 1e3))

        eps_history[it] = eps
        if bar is not None:
            bar.update(1)
            bar.set_postfix({"step_size": f"{eps:.2e}"})

    if bar is not None:
        bar.set_description("SGLD sampling")

    # ─── sampling (fixed step size) ──────────────────────────────────────
    n_iter_sample = n_samples * thin
    # Pre-allocate storage of u-samples for the kept iterations.
    u_samples = np.empty((n_chains, n_samples, D), dtype=np.float64)
    log_probs = np.empty((n_chains, n_samples), dtype=np.float64)

    keep_idx = 0
    for it in range(n_iter_sample):
        key, *sub = jr.split(key, n_chains + 1)
        chain_keys = jnp.stack(sub)
        if kind == "diag":
            us, logps, _ = step_fn(us, chain_keys, eps, precond_obj)
        else:
            us, logps, _ = step_fn(us, chain_keys, eps, precond_obj[0], precond_obj[1])

        if (it + 1) % thin == 0:
            u_samples[:, keep_idx] = np.asarray(us)
            log_probs[:, keep_idx] = np.asarray(logps)
            keep_idx += 1

        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()

    t1 = time.time()

    # ─── resolve constraints on samples ──────────────────────────────────
    samples_phys = _resolve_samples(problem_s, u_samples)             # dict[prefix -> (nc, ns, Nt)]

    # ─── diagnostics ────────────────────────────────────────────────────
    rhat_dict = {k: _rhat(v) for k, v in samples_phys.items()}
    ess_dict  = {k: _ess(v)  for k, v in samples_phys.items()}

    # Summary stats
    summary = {}
    for k, v in samples_phys.items():
        flat = v.reshape(-1, v.shape[-1])    # (nc*ns, Nt)
        # Robust correlation: zero-variance coords (constrained-to-constant
        # params like fixed Ti0) give 0/0 in np.corrcoef.
        if flat.shape[1] > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.corrcoef(flat.T)
            corr = np.where(np.isfinite(corr), corr, 0.0)
            np.fill_diagonal(corr, 1.0)
        else:
            corr = np.eye(1)
        summary[k] = {
            "mean": flat.mean(axis=0),
            "std":  flat.std(axis=0, ddof=1),
            "p16":  np.percentile(flat, 16, axis=0),
            "p50":  np.percentile(flat, 50, axis=0),
            "p84":  np.percentile(flat, 84, axis=0),
            "corr_intra": corr,
        }

    # max R-hat / min ESS report
    max_rhat = -np.inf
    max_rhat_key = ""
    min_ess = np.inf
    min_ess_key = ""
    for k, arr in rhat_dict.items():
        a = np.asarray(arr)
        if a.size and np.isfinite(a).any():
            i = int(np.nanargmax(a))
            if a[i] > max_rhat:
                max_rhat = float(a[i])
                max_rhat_key = f"{k}[t={i}]"
    for k, arr in ess_dict.items():
        a = np.asarray(arr)
        if a.size and np.isfinite(a).any():
            i = int(np.nanargmin(a))
            if a[i] < min_ess:
                min_ess = float(a[i])
                min_ess_key = f"{k}[t={i}]"

    return SimpleNamespace(
        samples_phys=samples_phys,
        u_samples=u_samples,
        log_probs=log_probs,
        summary=summary,
        rhat=rhat_dict,
        ess=ess_dict,
        step_size_history=eps_history,
        u_chain_init=np.asarray(u_chain_init),
        u_map=np.asarray(u_map),
        varying_keys=problem.varying_keys,
        prefixes=list(samples_phys),
        temperature=problem_s.temperature,
        n_pixels_valid=problem_s.n_pixels_valid,
        step_size_final=eps,
        n_chains=n_chains,
        n_samples=n_samples,
        burn_in=burn_in,
        thin=thin,
        precond=precond,
        seed=seed,
        max_rhat=max_rhat,
        max_rhat_key=max_rhat_key,
        min_ess=min_ess,
        min_ess_key=min_ess_key,
        wall_time=t1 - t0,
    )


# ─── helpers ────────────────────────────────────────────────────────────────

def _polish_map(problem_s, u_map, *, max_iter=200, tol=1e-9):
    """Short LBFGS on -target_log_prob to relocate to Jacobian-corrected mode."""
    import optax
    from collections import deque

    @jit
    def neg_logp_and_grad(u):
        v, g = problem_s.target_log_prob_and_grad(u)
        return -v, -g

    opt = optax.lbfgs()
    state = opt.init(u_map)

    @jit
    def step(u, state):
        val, grad_val = neg_logp_and_grad(u)
        updates, new_state = opt.update(
            grad_val, state, u,
            value=val, grad=grad_val,
            value_fn=lambda x: neg_logp_and_grad(x)[0],
        )
        u_new = optax.apply_updates(u, updates)
        return u_new, new_state, val

    u = u_map
    window = deque(maxlen=50)
    for _ in range(max_iter):
        u, state, val = step(u, state)
        window.append(float(val))
        if len(window) == window.maxlen:
            rel = (window[0] - min(window)) / (abs(window[0]) + 1e-300)
            if rel < tol:
                break
    return u


def _resolve_samples(problem_s, u_samples_np):
    """Apply constraint resolution to every u-sample.

    Returns dict[prefix -> (n_chains, n_samples, Nt) numpy array].
    """
    nc, ns, D = u_samples_np.shape
    flat = jnp.asarray(u_samples_np.reshape(nc * ns, D))

    resolved_one = problem_s.resolve_one
    resolved_vmap = jit(vmap(resolved_one))
    out = resolved_vmap(flat)
    # out: dict[prefix -> (nc*ns, Nt)]
    return {k: np.asarray(v).reshape(nc, ns, -1) for k, v in out.items()}
