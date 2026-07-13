"""Posterior samplers for Thomson-scattering fits: HMC / MALA / SGLD + Laplace.

Builds on :func:`ThomsonScattering.fitting._build_grad_problem`: takes the
``SimpleNamespace`` it returns and produces a JAX-jitted, Jacobian-corrected
sampling target plus a multi-chain MCMC runner with R-hat/ESS diagnostics
and per-sample constraint resolution.

Public API
----------
- :func:`build_sampling_problem` — wrap a fit problem with a sampling target.
- :func:`run_mcmc_posterior` — top-level multi-chain sampler (HMC/MALA/SGLD).
- :func:`run_sgld_posterior` — backward-compatible alias (``kernel="sgld"``).
- :func:`run_laplace_posterior` — Hessian-only error bars (no chains).

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

Kernels
-------
- ``hmc`` (default): leapfrog trajectories with the Hessian preconditioner as
  inverse mass matrix, Metropolis-corrected, dual-averaging step-size
  adaptation on the acceptance rate. Trajectory length is jittered uniformly
  in [1, n_leapfrog] each iteration to avoid resonances. Non-finite
  trajectory energies are rejected and counted as divergences.
- ``mala``: HMC with a single leapfrog step (Metropolis-adjusted Langevin).
- ``sgld``: the legacy unadjusted Langevin kernel with the drift/noise-ratio
  Robbins-Monro step adaptation. Kept for reproducibility of old runs; its
  samples carry an O(step_size) discretization bias that the Metropolis
  kernels do not.

Rolling our own kernels (vs. blackjax) keeps full control over the bijector
Jacobian and adds no new dependency.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import jax as _jax
_jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import jit, lax, value_and_grad, vmap, pmap
from jax.nn import log_sigmoid, sigmoid

from .parallel import serial_requested


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


def _make_sampling_bijector(problem):
    """Sampler-side unconstrained transform: logistic for two-sided bounds.

    The FIT uses lmfit's arcsin transform for two-sided bounds — fine for
    LBFGS, but *periodic* in u, and its Jacobian term log|cos u| puts -inf
    walls at u = ±π/2 (the x-bounds). A weakly identified bounded parameter
    (e.g. an inactive constraint dummy like ``ifract1_floor``) diffuses along
    its flat direction, keeps hitting those walls (unbounded gradients →
    step size collapses for every coordinate), and can even leapfrog across
    onto another sin branch. The sampler therefore re-parametrizes two-sided
    bounds with the logistic bijector

        x = lo + (hi - lo)·sigmoid(u),
        log|dx/du| = log(hi-lo) + log_sigmoid(u) + log_sigmoid(-u),

    which is monotone, aperiodic, and has d(log-jac)/du = 1 - 2·sigmoid(u)
    bounded in (-1, 1) — no walls, exponential (not singular) tails, and its
    implicit u-space prior is a well-scaled logistic bell (σ ≈ 1.8). The fit
    itself is untouched; only the sampling coordinates change. One-sided and
    unbounded coordinates reuse the fit's forms (already smooth/aperiodic).

    Returns ``(to_x, log_det_jac, to_u_np)``.
    """
    lo, hi = problem.lower, problem.upper
    lo_f, hi_f = jnp.isfinite(lo), jnp.isfinite(hi)
    two_sided = lo_f & hi_f
    lo_only = lo_f & ~hi_f
    hi_only = ~lo_f & hi_f
    lo_s = jnp.where(lo_f, lo, 0.0)
    hi_s = jnp.where(hi_f, hi, 1.0)
    width = jnp.where(two_sided, hi_s - lo_s, 1.0)

    def to_x(u):
        x_b = lo_s + width * sigmoid(u)
        x_l = lo_s - 1.0 + jnp.sqrt(u ** 2 + 1.0)
        x_h = hi_s + 1.0 - jnp.sqrt(u ** 2 + 1.0)
        return jnp.where(two_sided, x_b,
                         jnp.where(lo_only, x_l,
                                   jnp.where(hi_only, x_h, u)))

    def log_det_jac(u):
        lj_b = jnp.log(width) + log_sigmoid(u) + log_sigmoid(-u)
        one_sided = jnp.log(jnp.maximum(jnp.abs(u), 1e-30)) \
            - 0.5 * jnp.log(u ** 2 + 1.0)
        lj = jnp.where(two_sided, lj_b,
                       jnp.where(lo_only | hi_only, one_sided, 0.0))
        return jnp.sum(lj)

    lo_np, hi_np = np.asarray(problem.lower_np), np.asarray(problem.upper_np)

    def to_u_np(x):
        x = np.asarray(x, dtype=np.float64)
        u = np.array(x)
        for i in range(x.shape[0]):
            l, h = lo_np[i], hi_np[i]
            if np.isfinite(l) and np.isfinite(h):
                frac = np.clip((x[i] - l) / (h - l), 1e-12, 1.0 - 1e-12)
                u[i] = np.log(frac) - np.log1p(-frac)
            elif np.isfinite(l):
                u[i] = np.sqrt(max((x[i] - l + 1.0) ** 2 - 1.0, 0.0))
            elif np.isfinite(h):
                u[i] = np.sqrt(max((h - x[i] + 1.0) ** 2 - 1.0, 0.0))
        return u

    return to_x, log_det_jac, to_u_np


def build_sampling_problem(problem, *, temperature=None):
    """Wrap a fit problem with a Jacobian-corrected, temperature-rescaled
    sampling target.

    The target lives in the sampler's own unconstrained coordinates (see
    :func:`_make_sampling_bijector` — logistic for two-sided bounds, the
    fit's transforms elsewhere), NOT the fit's arcsin coordinates. Callers
    that hold the fit-space MAP should convert with
    ``u_s = problem_s.u_from_fit(u_fit)``.

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
        to_x(u) / u_from_x_np(x) / u_from_fit(u_fit)
        temperature : float
        n_pixels_valid : int
    """
    T = _resolve_temperature(temperature, problem)

    mask = jnp.isnan(problem.Pkl_data) | jnp.isnan(problem.Pkl_var)
    n_pix = int(jnp.sum(~mask))

    to_x, log_det_jac, to_u_np = _make_sampling_bijector(problem)

    def target_log_prob(u):
        return -problem.objective_flat(to_x(u)) / T + log_det_jac(u)

    log_det_jac_jit = jit(log_det_jac)
    target_log_prob_jit = jit(target_log_prob)
    target_log_prob_and_grad_jit = jit(value_and_grad(target_log_prob))

    @jit
    def resolve_one(u):
        return problem.build_params_dict(to_x(u))

    def u_from_fit(u_fit):
        """Convert a FIT-space (arcsin) unconstrained vector to sampling
        coordinates via physical space."""
        x = np.asarray(problem.to_external_jax(jnp.asarray(u_fit)))
        return to_u_np(x)

    return SimpleNamespace(
        problem=problem,
        temperature=T,
        n_pixels_valid=n_pix,
        log_det_jac=log_det_jac_jit,
        target_log_prob=target_log_prob_jit,
        target_log_prob_and_grad=target_log_prob_and_grad_jit,
        resolve_one=resolve_one,
        to_x=jit(to_x),
        u_from_x_np=to_u_np,
        u_from_fit=u_from_fit,
    )


# ─── Hessian machinery ──────────────────────────────────────────────────────

def _batched_grads(problem_s, pts, *, batch_size=16):
    """Gradients of ``target_log_prob`` at a stack of points.

    pts : (N, D) → (N, D). Evaluated with ``lax.map(..., batch_size=)`` so the
    points are vmapped in chunks: bounded memory, but no per-point Python
    dispatch (the old FD loop paid one jitted call per point, which dominated
    on the general-distribution path).
    """
    grad_fn = lambda u: problem_s.target_log_prob_and_grad(u)[1]
    mapped = jit(lambda xs: lax.map(grad_fn, xs, batch_size=batch_size))
    return mapped(jnp.asarray(pts))


def _full_hessian_fd(problem_s, u_ref, *, h=1e-4, batch_size=16):
    """Full Hessian via central finite differences of the gradient.

    Avoids JAX's 2nd-derivative rules entirely — works when the forward
    model contains operators whose Hessian isn't implemented (e.g.
    ``gammaincc`` w.r.t. its first argument, which trips a
    ``NotImplementedError: igamma_grad_a`` when ``pe``/``pi`` are free).

    Cost: 2·D gradient evaluations, batched (see :func:`_batched_grads`).
    """
    u_ref = jnp.asarray(u_ref)
    D = int(u_ref.shape[0])
    eye_h = jnp.eye(D, dtype=u_ref.dtype) * h
    pts = jnp.concatenate([u_ref[None, :] + eye_h, u_ref[None, :] - eye_h])
    G = _batched_grads(problem_s, pts, batch_size=batch_size)     # (2D, D)
    H = (G[:D] - G[D:]).T / (2 * h)                               # column j = ∂g/∂u_j
    return jnp.asarray(0.5 * (H + H.T))


def _diag_hessian_fd(problem_s, u_ref, *, h=1e-4, batch_size=16):
    """Diagonal of the finite-difference Hessian (same 2·D gradient cost)."""
    return jnp.diag(_full_hessian_fd(problem_s, u_ref, h=h,
                                     batch_size=batch_size))


def _prefer_fd_hessian(problem_s):
    """True when the problem contains a general-path (quadrature) model.

    ``jax.hessian`` through the quadrature's ``lax.map`` is technically
    differentiable but compiles forward-over-reverse through the whole
    time-mapped kernel — prohibitively slow. The finite-difference fallback
    (2·D extra gradient evals) is far cheaper there.
    """
    from .distributions import GeneralDistribution
    prob = getattr(problem_s, "problem", None)
    models = (tuple(getattr(prob, "e_models", ()))
              + tuple(getattr(prob, "i_models", ())))
    return any(isinstance(m, GeneralDistribution) for m in models)


def _full_hessian(problem_s, u_ref):
    """Full Hessian of ``target_log_prob`` at ``u_ref`` (analytic; FD fallback).

    This is the Hessian of the *temperature-scaled, Jacobian-corrected*
    log-posterior, so ``-inv(H)`` is the Laplace covariance in u-space and the
    eigenvalues should be negative at a well-formed MAP. Symmetrized and
    NaN-cleaned. Falls back to (batched) finite differences when JAX lacks a
    2nd-derivative rule for the forward model, or skips the analytic attempt
    outright when any species uses the general quadrature path (where the
    analytic Hessian's compile time is prohibitive).
    """
    if _prefer_fd_hessian(problem_s):
        print("  [hessian] general-path distribution present; "
              "using finite-difference Hessian.")
        H = _full_hessian_fd(problem_s, u_ref)
    else:
        try:
            H = _jax.hessian(problem_s.target_log_prob)(u_ref)
        except NotImplementedError as err:
            print(f"  [hessian] analytical Hessian unavailable ({err}); "
                  f"falling back to finite-difference Hessian.")
            H = _full_hessian_fd(problem_s, u_ref)
    H = jnp.where(jnp.isfinite(H), H, 0.0)
    return 0.5 * (H + H.T)


def _curvature_floors(problem, *, floor=1e-6):
    """Per-coordinate lower bound on the |Hessian| used in preconditioners.

    A two-sided bounded parameter lives on the sampler's logistic bijector:
    even a completely flat direction (e.g. an inactive constraint dummy like
    ``ifract1_floor``) has the implicit logistic u-prior, variance π²/3 ≈ 3.3
    — so its preconditioner variance is capped at 4 by flooring the
    curvature at 1/4. Without this, a direction that is flat at the MAP gets
    curvature ~0 → preconditioner variance up to 1/floor = 1e6 → chains are
    initialized and kicked absurd distances along it, and the step size
    collapses for every other coordinate. One-sided / unbounded coordinates
    keep the generic ``floor``.
    """
    lo, hi = np.asarray(problem.lower_np), np.asarray(problem.upper_np)
    bounded = np.isfinite(lo) & np.isfinite(hi)
    return jnp.asarray(np.where(bounded, 0.25, floor))


def _build_diag_hessian_precond(problem_s, u_ref, *, floor=1e-6,
                                fallback=1.0, H=None):
    """Diagonal preconditioner from |H(target_log_prob)(u_ref)|.

    Returns ``M_diag`` of shape (D,) with
    ``M_diag[i] = 1 / max(|H_ii|, floor_i)`` where ``floor_i`` comes from
    :func:`_curvature_floors` (bounded coordinates are capped at (π/2)²
    variance; others at 1/floor).

    If ``H`` (a precomputed full Hessian at ``u_ref``) is given, its diagonal
    is reused instead of recomputing; otherwise the Hessian is obtained via
    :func:`_full_hessian`. Non-finite entries are replaced with ``fallback``
    so a single bad coordinate doesn't NaN the entire preconditioner.
    """
    if H is None:
        H = _full_hessian(problem_s, u_ref)
    floors = _curvature_floors(problem_s.problem, floor=floor)
    h_diag = jnp.abs(jnp.diag(H))
    h_diag = jnp.where(jnp.isfinite(h_diag), h_diag, fallback)
    M = 1.0 / jnp.maximum(h_diag, floors)
    return jnp.where(jnp.isfinite(M), M, 1.0)


def _build_full_hessian_precond(problem_s, u_ref, *, reg=1e-6, H=None):
    """Full Hessian preconditioner ``M = (|H| + diag(floors))^{-1}``.

    Builds an SPD approximation by taking the absolute eigenvalues (so the
    preconditioner is well-defined even where the Hessian has saddle-point
    directions), then adds the per-coordinate curvature floors from
    :func:`_curvature_floors` (capping flat bounded directions at (π/2)²
    variance) before inverting. Returns ``(M, L_chol, L_mass)``:

    - ``M``       : the preconditioner (≈ posterior covariance in u-space);
    - ``L_chol``  : ``L_chol L_chol^T = M`` — SGLD's noise factor;
    - ``L_mass``  : ``L_mass L_mass^T = M^{-1}`` — HMC's momentum-draw factor
      (the mass matrix is ``M^{-1}``).

    If ``H`` (a precomputed full Hessian at ``u_ref``) is given, it is reused
    instead of recomputing.
    """
    if H is None:
        H = _full_hessian(problem_s, u_ref)
    floors = _curvature_floors(problem_s.problem, floor=reg)
    w, V = jnp.linalg.eigh(0.5 * (H + H.T))
    A = (V * jnp.maximum(jnp.abs(w), 0.0)) @ V.T + jnp.diag(floors) + reg * jnp.eye(H.shape[0])
    w2, V2 = jnp.linalg.eigh(0.5 * (A + A.T))
    w2 = jnp.maximum(w2, reg)
    M = (V2 * (1.0 / w2)) @ V2.T
    L_chol = (V2 * (1.0 / jnp.sqrt(w2))) @ V2.T
    L_mass = (V2 * jnp.sqrt(w2)) @ V2.T
    return M, L_chol, L_mass


# ─── Laplace (delta-method) physical covariance ─────────────────────────────

def _laplace_physical_cov(problem_s, u_ref, hessian_u, *, tol=1e-8,
                          n_loadings=4):
    """Delta-method physical-parameter covariance from the MAP Hessian.

    With ``Σ_u = -inv(H)`` the Laplace covariance in u-space and
    ``J = ∂(physical)/∂u`` (Jacobian of ``resolve_one``), the physical
    covariance is ``Σ_phys = J Σ_u Jᵀ`` (P×P). It is rank ≤ D — singular along
    the constraints (tied species, the simplex remainder) — so it has no full
    inverse, but its diagonal gives valid 1σ physical error bars and any
    sub-block/linear combo is well-defined.

    Eigenvalues of H are negative at a maximum; any ``> -tol·|λ|max`` flag a
    non-identified direction (infinite u-space variance). Those are dropped
    from ``Σ_u`` (so they don't blow the matrix up), counted, and *reported*:
    each flat eigenvector is projected through J and normalized by the MAP
    parameter magnitudes, so the report names the physical parameter
    combination that the data cannot pin down (e.g. a p↔Te trade-off).

    Returns a SimpleNamespace:
        cov_phys        : (P, P) ndarray
        labels          : list[str], "<prefix>[t=k]" row/col order
        sigma_by_prefix : dict[prefix -> (Nt,) 1σ]
        n_nonidentified : int
        nonid_loadings  : (n_bad, P) physical relative loadings (unit rows)
        nonid_descriptions : list[str] human-readable top-loading summaries
    """
    u_ref = jnp.asarray(u_ref)
    d0 = problem_s.resolve_one(u_ref)
    prefixes = list(d0.keys())
    nts = {p: int(d0[p].shape[0]) for p in prefixes}

    def flat_resolve(u):
        d = problem_s.resolve_one(u)
        return jnp.concatenate([d[p] for p in prefixes])

    J = _jax.jacfwd(flat_resolve)(u_ref)                       # (P, D)
    w, V = jnp.linalg.eigh(0.5 * (hessian_u + hessian_u.T))    # ascending
    wmax = jnp.maximum(jnp.max(jnp.abs(w)), 1.0)
    bad = w > -tol * wmax                                       # non-negative curvature
    inv_var = jnp.where(bad, 0.0, -1.0 / w)                    # -1/λ for λ < 0
    Sigma_u = (V * inv_var) @ V.T
    Sigma_phys = J @ Sigma_u @ J.T
    Sigma_phys = np.asarray(0.5 * (Sigma_phys + Sigma_phys.T))

    sigma_flat = np.sqrt(np.clip(np.diag(Sigma_phys), 0.0, None))
    labels, sigma_by_prefix, i = [], {}, 0
    for p in prefixes:
        labels += [f"{p}[t={t}]" for t in range(nts[p])]
        sigma_by_prefix[p] = sigma_flat[i:i + nts[p]]
        i += nts[p]

    # Non-identified directions, expressed in physical space. J maps a u-space
    # eigenvector to absolute physical shifts; dividing by the MAP magnitudes
    # gives comparable *relative* loadings across parameters with wildly
    # different units (n ~ 1e20 vs ifract ~ 1).
    bad_np = np.asarray(bad)
    nonid_loadings = np.zeros((0, len(labels)))
    nonid_descriptions = []
    if bad_np.any():
        x_flat = np.concatenate([np.asarray(d0[p]) for p in prefixes])
        scale = np.maximum(np.abs(x_flat), 1e-30)
        J_np, V_np = np.asarray(J), np.asarray(V)
        rows = []
        for idx in np.nonzero(bad_np)[0]:
            d_rel = (J_np @ V_np[:, idx]) / scale
            nrm = np.linalg.norm(d_rel)
            d_rel = d_rel / nrm if nrm > 0 else d_rel
            rows.append(d_rel)
            order = np.argsort(-np.abs(d_rel))[:n_loadings]
            terms = [f"{d_rel[j]:+.2f}·{labels[j]}"
                     for j in order if abs(d_rel[j]) > 0.05]
            nonid_descriptions.append(" ".join(terms) if terms
                                      else "(no physical projection — "
                                           "direction lies in a constrained "
                                           "null space)")
        nonid_loadings = np.stack(rows) if rows else nonid_loadings

    return SimpleNamespace(
        cov_phys=Sigma_phys,
        labels=labels,
        sigma_by_prefix=sigma_by_prefix,
        n_nonidentified=int(bad_np.sum()),
        nonid_loadings=nonid_loadings,
        nonid_descriptions=nonid_descriptions,
    )


def _print_nonidentified(lap):
    if lap.n_nonidentified:
        print(f"  [laplace] {lap.n_nonidentified} non-identified direction(s) "
              f"at the MAP (non-negative Hessian curvature); dropped from "
              f"Σ_phys. Flat physical directions (relative loadings):")
        for k, desc in enumerate(lap.nonid_descriptions):
            print(f"    [{k}] {desc}")


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


# ─── HMC / MALA step kernels ────────────────────────────────────────────────

def _hmc_step(u, logp, g, key, eps, n_leap, mv_M, draw_p, kinetic,
              target_grad_fn):
    """One HMC iteration: momentum draw → leapfrog(n_leap) → Metropolis.

    ``mv_M(p)`` applies the preconditioner (inverse mass) to a momentum,
    ``draw_p(xi)`` maps a standard normal to p ~ N(0, M⁻¹), and
    ``kinetic(p) = ½ pᵀ M p``. The carried ``(logp, g)`` at ``u`` avoid a
    fresh gradient at the start of every trajectory, so an iteration costs
    exactly ``n_leap`` gradient evaluations.

    Non-finite trajectory energy ⇒ divergence: the proposal is rejected and
    flagged. Returns ``(u', logp', g', accept_prob, accepted, divergent)``.
    """
    key_mom, key_acc = jr.split(key)
    xi = jr.normal(key_mom, u.shape, dtype=u.dtype)
    p0 = draw_p(xi)
    K0 = kinetic(p0)

    def body(i, carry):
        u_c, p_c, logp_c, g_c = carry
        u_n = u_c + eps * mv_M(p_c)
        logp_n, g_n = target_grad_fn(u_n)
        p_n = p_c + eps * g_n
        return (u_n, p_n, logp_n, g_n)

    # Standard leapfrog with the half-steps folded in: seed p with +eps/2·g(u0),
    # run n_leap full (u, p) updates, then remove the surplus eps/2·g(u_L).
    p_seed = p0 + 0.5 * eps * g
    u_f, p_f, logp_f, g_f = lax.fori_loop(
        0, n_leap, body, (u, p_seed, logp, g))
    p_f = p_f - 0.5 * eps * g_f
    K_f = kinetic(p_f)

    log_ratio = (logp_f - K_f) - (logp - K0)
    divergent = ~jnp.isfinite(log_ratio)
    log_ratio = jnp.where(divergent, -jnp.inf, log_ratio)
    accept_prob = jnp.minimum(1.0, jnp.exp(log_ratio))
    accepted = jnp.log(jr.uniform(key_acc, (), dtype=u.dtype)) < log_ratio
    u_out = jnp.where(accepted, u_f, u)
    logp_out = jnp.where(accepted, logp_f, logp)
    g_out = jnp.where(accepted, g_f, g)
    return u_out, logp_out, g_out, accept_prob, accepted, divergent


def _make_hmc_one_step(problem_s, kind, precond_obj):
    """Bind the preconditioner into a per-chain HMC step function."""
    tg = problem_s.target_log_prob_and_grad
    if kind == "diag":
        M_diag = precond_obj
        inv_sqrt_M = 1.0 / jnp.sqrt(M_diag)
        mv_M = lambda p: M_diag * p
        draw_p = lambda xi: inv_sqrt_M * xi
        kinetic = lambda p: 0.5 * jnp.sum(M_diag * p * p)
    else:
        M_full, _L_chol, L_mass = precond_obj
        mv_M = lambda p: M_full @ p
        draw_p = lambda xi: L_mass @ xi
        kinetic = lambda p: 0.5 * jnp.dot(p, M_full @ p)

    def one_step(u, logp, g, key, eps, n_leap):
        return _hmc_step(u, logp, g, key, eps, n_leap,
                         mv_M, draw_p, kinetic, tg)
    return one_step


# ─── dual-averaging step-size adaptation (Hoffman & Gelman 2014) ────────────

_DA_GAMMA, _DA_T0, _DA_KAPPA = 0.05, 10.0, 0.75
_LOG_EPS_MIN, _LOG_EPS_MAX = np.log(1e-12), np.log(1e3)


def _da_init(eps0):
    """(t, H_bar, log_eps, log_eps_bar) — log_eps_bar is the final answer."""
    log_e = jnp.log(eps0)
    return (jnp.zeros(()), jnp.zeros(()), log_e, log_e)


def _da_update(state, alpha_mean, mu, target):
    t, H_bar, log_eps, log_eps_bar = state
    t = t + 1.0
    H_bar = (1.0 - 1.0 / (t + _DA_T0)) * H_bar \
        + (target - alpha_mean) / (t + _DA_T0)
    log_eps = jnp.clip(mu - jnp.sqrt(t) / _DA_GAMMA * H_bar,
                       _LOG_EPS_MIN, _LOG_EPS_MAX)
    eta = t ** (-_DA_KAPPA)
    log_eps_bar = eta * log_eps + (1.0 - eta) * log_eps_bar
    return (t, H_bar, log_eps, log_eps_bar)


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

def _rank_normalize(x):
    """Rank-normalize pooled draws per coordinate (Vehtari et al. 2021).

    x : (m, n, *) — average-rank over the pooled m·n draws, mapped through the
    normal quantile function with the (r - 3/8)/(S + 1/4) offset.
    """
    from scipy.special import ndtri
    m, n = x.shape[0], x.shape[1]
    flat = x.reshape(m * n, *x.shape[2:])
    # argsort-of-argsort ranks (0-based); ties are broken by order, which is
    # immaterial for continuous MCMC draws.
    order = np.argsort(flat, axis=0)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order,
                      np.arange(m * n).reshape(-1, *([1] * (flat.ndim - 1)))
                      * np.ones_like(order), axis=0)
    z = ndtri((ranks + 1 - 0.375) / (m * n + 0.25))
    return z.reshape(m, n, *x.shape[2:])


def _rhat_basic(x):
    """Plain Gelman-Rubin R-hat on (m, n, *) chains."""
    m, n = x.shape[0], x.shape[1]
    chain_means = x.mean(axis=1)                                    # (m, *)
    B = n * np.var(chain_means, axis=0, ddof=1)
    W = np.mean(np.var(x, axis=1, ddof=1), axis=0)
    var_hat = ((n - 1) / n) * W + B / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_hat / np.maximum(W, 1e-300))


def _rhat(samples):
    """Split-chain, rank-normalized R-hat (Vehtari et al. 2021).

    samples : array of shape (n_chains, n_samples, *)
    Each chain is split in half (catching within-chain drift that plain R-hat
    misses) and the pooled draws are rank-normalized before the classic
    formula, making the diagnostic robust to heavy tails. Returns the max of
    the bulk (rank-normalized) and naive R-hat per coordinate. Coordinates
    whose samples are effectively constant (relative std < 1e-10) return NaN.
    """
    x = np.asarray(samples)
    nc, ns = x.shape[0], x.shape[1]
    if nc < 2 or ns < 4:
        return np.full(x.shape[2:], np.nan)
    half = ns // 2
    xs = np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)  # (2nc, half, *)
    rhat = np.maximum(_rhat_basic(_rank_normalize(xs)), _rhat_basic(xs))
    # Mask coordinates whose samples are effectively constant.
    grand_mean = x.reshape(-1, *x.shape[2:]).mean(axis=0)
    overall_std = x.reshape(-1, *x.shape[2:]).std(axis=0)
    scale = np.maximum(np.abs(grand_mean), 1.0)
    return np.where(overall_std > 1e-10 * scale, rhat, np.nan)


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


# ─── chunked scan runners ───────────────────────────────────────────────────
#
# All kernels advance through jitted `lax.scan` chunks (default ~100
# iterations per chunk) instead of one jitted call per iteration: the
# adaptation math lives on-device inside the scan, so the only host↔device
# traffic is one carry + output transfer per chunk (the old loop paid a
# Python dispatch and a host sync every single step). The outer Python loop
# over chunks drives tqdm and bounds compile time. With >1 device and one
# chain per device the same scan bodies run inside `pmap` (scan-inside-pmap),
# with cross-chain reductions via `lax.pmean`/`lax.all_gather`; shared state
# (step size, adaptation) is computed identically on every device.

def _split_chunks(total, chunk_size):
    sizes = [chunk_size] * (total // chunk_size)
    if total % chunk_size:
        sizes.append(total % chunk_size)
    return sizes


def _chunk_keys(key, size, n_chains):
    """Host-side per-iteration per-chain keys for one chunk.

    Returns (key', keys) with keys shaped (size, n_chains, 2).
    """
    key, sub = jr.split(key)
    keys = jr.split(sub, size * n_chains).reshape(size, n_chains, 2)
    return key, keys


def _chunk_lkeys(key, size):
    """Per-iteration keys for the trajectory-length draw (shared across
    chains — lockstep vmap/pmap requires one L per iteration)."""
    key, sub = jr.split(key)
    return key, jr.split(sub, size).reshape(size, 2)


def _draw_n_leap(key_L, eps, traj_length, n_leap_cap):
    """Jittered leapfrog count for one iteration: uniform in [1, L_hi] with
    ``L_hi = clip(ceil(traj_length / eps), 1, n_leap_cap)``.

    Holding the *trajectory length* ``eps·L`` (in preconditioned-σ units)
    roughly constant is what makes HMC robust to whatever step size dual
    averaging settles on: when curvature (e.g. a constraint kink) forces eps
    down, L rises to compensate, so proposals still travel O(1) posterior
    widths instead of degenerating into a random walk. The uniform jitter
    avoids periodic-orbit resonances.
    """
    L_hi = jnp.clip(jnp.ceil(traj_length / eps).astype(jnp.int32),
                    1, n_leap_cap)
    # Jitter over the upper half of [1, L_hi]: enough spread to kill
    # resonances without wasting half the gradient budget on short hops.
    L_lo = jnp.maximum(1, L_hi // 2)
    return jr.randint(key_L, (), L_lo, L_hi + 1)


def _make_hmc_chunk_fns(problem_s, kind, precond_obj, n_chains, mu, target,
                        traj_length, n_leap_cap, use_pmap, devices):
    """Build jitted (burn_chunk, samp_chunk) for the HMC/MALA kernel.

    burn_chunk(us, logps, gs, da, keys, lkeys)
        -> us, logps, gs, da, (eps_hist, alpha_hist, div_hist, nl_hist)
    samp_chunk(us, logps, gs, eps, keys, lkeys)
        -> us, logps, gs, (u_hist, logp_hist, acc_hist, div_hist, nl_hist)

    The per-iteration leapfrog count is drawn on-device from the *current*
    eps (see :func:`_draw_n_leap`), shared across chains (lockstep).
    """
    one_step = _make_hmc_one_step(problem_s, kind, precond_obj)

    if not use_pmap:
        step_c = vmap(one_step, in_axes=(0, 0, 0, 0, None, None))

        def burn_chunk(us, logps, gs, da, keys, lkeys):
            def body(carry, xs):
                us, logps, gs, da = carry
                k, kl = xs
                eps = jnp.exp(da[2])
                nl = _draw_n_leap(kl, eps, traj_length, n_leap_cap)
                us, logps, gs, aprob, acc, div = step_c(us, logps, gs, k, eps, nl)
                alpha = jnp.mean(aprob)
                da = _da_update(da, alpha, mu, target)
                return (us, logps, gs, da), (eps, alpha, jnp.sum(div), nl)
            (us, logps, gs, da), hist = lax.scan(
                body, (us, logps, gs, da), (keys, lkeys))
            return us, logps, gs, da, hist

        def samp_chunk(us, logps, gs, eps, keys, lkeys):
            def body(carry, xs):
                us, logps, gs = carry
                k, kl = xs
                nl = _draw_n_leap(kl, eps, traj_length, n_leap_cap)
                us, logps, gs, aprob, acc, div = step_c(us, logps, gs, k, eps, nl)
                return (us, logps, gs), (us, logps, acc, div, nl)
            (us, logps, gs), hist = lax.scan(
                body, (us, logps, gs), (keys, lkeys))
            return us, logps, gs, hist

        return jit(burn_chunk), jit(samp_chunk)

    # pmap backend: one chain per device, scan inside pmap. Shared quantities
    # (eps, dual-averaging state, the L draw) are derived from pmean'd
    # acceptance and a broadcast L-key, so every device computes identical
    # copies; the host reads device 0.
    def dev_burn(u, logp, g, da, keys_dev, lkeys):
        def body(carry, xs):
            u, logp, g, da = carry
            k, kl = xs
            eps = jnp.exp(da[2])
            nl = _draw_n_leap(kl, eps, traj_length, n_leap_cap)
            u, logp, g, aprob, acc, div = one_step(u, logp, g, k, eps, nl)
            alpha = lax.pmean(aprob, axis_name="chains")
            da = _da_update(da, alpha, mu, target)
            div_tot = lax.psum(div.astype(jnp.int32), axis_name="chains")
            return (u, logp, g, da), (eps, alpha, div_tot, nl)
        (u, logp, g, da), hist = lax.scan(
            body, (u, logp, g, da), (keys_dev, lkeys))
        return u, logp, g, da, hist

    def dev_samp(u, logp, g, eps, keys_dev, lkeys):
        def body(carry, xs):
            u, logp, g = carry
            k, kl = xs
            nl = _draw_n_leap(kl, eps, traj_length, n_leap_cap)
            u, logp, g, aprob, acc, div = one_step(u, logp, g, k, eps, nl)
            return (u, logp, g), (u, logp, acc, div, nl)
        (u, logp, g), hist = lax.scan(body, (u, logp, g), (keys_dev, lkeys))
        return u, logp, g, hist

    burn_p = pmap(dev_burn, axis_name="chains",
                  in_axes=(0, 0, 0, None, 1, None), devices=devices)
    samp_p = pmap(dev_samp, axis_name="chains",
                  in_axes=(0, 0, 0, None, 1, None), devices=devices)

    def burn_chunk(us, logps, gs, da, keys, lkeys):
        us, logps, gs, da_r, hist = burn_p(us, logps, gs, da, keys, lkeys)
        # da_r / eps_hist / alpha_hist replicated across devices; take dev 0.
        da = tuple(v[0] for v in da_r)
        eps_h, alpha_h, div_h, nl_h = hist
        return us, logps, gs, da, (eps_h[0], alpha_h[0], div_h[0], nl_h[0])

    def samp_chunk(us, logps, gs, eps, keys, lkeys):
        us, logps, gs, hist = samp_p(us, logps, gs, eps, keys, lkeys)
        u_h, logp_h, acc_h, div_h, nl_h = hist    # (nc, size, ...) — chain-major
        return us, logps, gs, (jnp.moveaxis(u_h, 0, 1),
                               jnp.moveaxis(logp_h, 0, 1),
                               jnp.moveaxis(acc_h, 0, 1),
                               jnp.moveaxis(div_h, 0, 1),
                               nl_h[0])
    return burn_chunk, samp_chunk


def _make_sgld_chunk_fns(problem_s, kind, precond_obj, n_chains, burn_in,
                         adapt_step, adapt_target, rmsprop, use_pmap, devices):
    """Build jitted (burn_chunk, samp_chunk) for the legacy SGLD kernel.

    burn_chunk(us, eps, rms, it0, keys) -> us, eps, rms, (eps_hist,)
    samp_chunk(us, eps, M_diag_or_none, keys) -> us, (u_hist, logp_hist)

    The Robbins-Monro drift/noise adaptation and (optionally) the RMSProp
    preconditioner EMA run on-device inside the scan; `it0` carries the global
    burn-in iteration index for the decaying learning rate.
    """
    tg = problem_s.target_log_prob_and_grad
    rmsprop_beta, rmsprop_eps = 0.95, 1e-6
    tgt = max(float(adapt_target), 1e-3)

    if kind == "diag":
        def one_step(u, key, eps, M_diag):
            return _sgld_step_diag(u, key, eps, M_diag, tg)
    else:
        M_full, L_chol, _L_mass = precond_obj

        def one_step(u, key, eps, _M_unused):
            return _sgld_step_full(u, key, eps, M_full, L_chol, tg)

    if not use_pmap:
        step_c = vmap(one_step, in_axes=(0, 0, None, None))

        def _median_ratio(gs, eps, M_diag):
            M_r = M_diag if kind == "diag" else M_full
            return _drift_noise_ratio(gs, eps, M_r, kind)
    else:
        step_c = None  # per-device one_step used directly below

    M_static = precond_obj if kind == "diag" else jnp.zeros(())

    if not use_pmap:
        def burn_chunk(us, eps, rms, it0, keys):
            def body(carry, k):
                us, eps, rms, it = carry
                M_diag = (1.0 / (jnp.sqrt(rms) + rmsprop_eps)) if rmsprop \
                    else M_static
                us, logps, gs = step_c(us, k, eps, M_diag)
                if rmsprop:
                    g_sq = jnp.mean(gs ** 2, axis=0)
                    rms = rmsprop_beta * rms + (1 - rmsprop_beta) * g_sq
                if adapt_step:
                    r_t = _median_ratio(gs, eps, M_diag)
                    eta = 0.05 / (1.0 + 10.0 * (it / max(1, burn_in)))
                    eps = eps * jnp.exp(eta * (adapt_target - r_t) / tgt)
                    eps = jnp.clip(eps, 1e-12, 1e3)
                return (us, eps, rms, it + 1.0), eps
            (us, eps, rms, it0), eps_hist = lax.scan(
                body, (us, eps, rms, it0), keys)
            return us, eps, rms, it0, eps_hist

        def samp_chunk(us, eps, M_diag, keys):
            def body(carry, k):
                us = carry
                us, logps, gs = step_c(us, k, eps, M_diag)
                return us, (us, logps)
            us, hist = lax.scan(body, us, keys)
            return us, hist

        return jit(burn_chunk), jit(samp_chunk)

    # pmap backend (one chain per device). The median drift/noise ratio needs
    # every chain's gradients: all_gather them so each device computes the
    # identical global median and hence identical eps updates.
    def dev_burn(u, eps, rms, it0, keys_dev):
        def body(carry, k):
            u, eps, rms, it = carry
            M_diag = (1.0 / (jnp.sqrt(rms) + rmsprop_eps)) if rmsprop \
                else M_static
            u, logp, g = one_step(u, k, eps, M_diag)
            if rmsprop:
                g_sq = lax.pmean(g ** 2, axis_name="chains")
                rms = rmsprop_beta * rms + (1 - rmsprop_beta) * g_sq
            if adapt_step:
                gs = lax.all_gather(g, axis_name="chains")   # (nc, D)
                M_r = M_diag if kind == "diag" else M_full
                r_t = _drift_noise_ratio(gs, eps, M_r, kind)
                eta = 0.05 / (1.0 + 10.0 * (it / max(1, burn_in)))
                eps = eps * jnp.exp(eta * (adapt_target - r_t) / tgt)
                eps = jnp.clip(eps, 1e-12, 1e3)
            return (u, eps, rms, it + 1.0), eps
        (u, eps, rms, it0), eps_hist = lax.scan(
            body, (u, eps, rms, it0), keys_dev)
        return u, eps, rms, it0, eps_hist

    def dev_samp(u, eps, M_diag, keys_dev):
        def body(carry, k):
            u = carry
            u, logp, g = one_step(u, k, eps, M_diag)
            return u, (u, logp)
        u, hist = lax.scan(body, u, keys_dev)
        return u, hist

    burn_p = pmap(dev_burn, axis_name="chains",
                  in_axes=(0, None, None, None, 1), devices=devices)
    samp_p = pmap(dev_samp, axis_name="chains",
                  in_axes=(0, None, None, 1), devices=devices)

    def burn_chunk(us, eps, rms, it0, keys):
        us, eps_r, rms_r, it_r, eps_hist = burn_p(us, eps, rms, it0, keys)
        return us, eps_r[0], rms_r[0], it_r[0], eps_hist[0]

    def samp_chunk(us, eps, M_diag, keys):
        us, hist = samp_p(us, eps, M_diag, keys)
        u_h, logp_h = hist                        # (nc, size, ...)
        return us, (jnp.moveaxis(u_h, 0, 1), jnp.moveaxis(logp_h, 0, 1))

    return burn_chunk, samp_chunk


# ─── top-level drivers ──────────────────────────────────────────────────────

_KERNELS = ("hmc", "mala", "sgld")


def run_mcmc_posterior(problem, u_map, *,
                       kernel="hmc",
                       temperature=None,
                       n_samples=1000, n_chains=4,
                       burn_in=None, thin=1, perturb_scale=1.0,
                       step_size=None, adapt_step=True, adapt_target=None,
                       traj_length=None, n_leapfrog=64,
                       precond="diag_hessian",
                       seed=0, progress=False,
                       polish_map=False, polish_max_iter=200,
                       chunk_size=100,
                       laplace_tol=1e-8):
    """Run multi-chain MCMC (HMC, MALA, or legacy SGLD).

    Parameters
    ----------
    problem : SimpleNamespace
        Output of :func:`ThomsonScattering.fitting._build_grad_problem`.
    u_map : array (D,)
        LBFGS MAP in unconstrained space (e.g. from running ``run_fit_grad``
        and re-encoding ``result.x`` via ``problem.to_internal_np``).
    kernel : {"hmc", "mala", "sgld"}
        - ``hmc`` (default): Metropolis-corrected leapfrog trajectories.
          Costs ~``n_leapfrog/2`` gradients per iteration but decorrelates
          far faster per iteration, and its error bars carry no step-size
          bias. Best for correlated/degenerate posteriors.
        - ``mala``: HMC with one leapfrog step. Cheapest exact kernel.
        - ``sgld``: legacy unadjusted Langevin (biased at finite step size).
    temperature : float, ``"auto"``, ``"unit"``, or None
    n_samples : int
        Number of post-burn-in, post-thin samples per chain.
    n_chains : int
        Independent chains; init at u_map + perturb_scale * N(0, I).
    burn_in : int or None
        Burn-in iterations per chain. Default = n_samples.
    thin : int
        Keep every ``thin``-th sample after burn-in.
    step_size : float or None
        Initial step size in u-space. Defaults: 0.5 for hmc/mala (the
        preconditioner makes the posterior ≈ unit-scale, where leapfrog is
        stable up to eps ≈ 2), 0.1 for sgld. Adapted during burn-in when
        ``adapt_step`` is True.
    adapt_step : bool
        hmc/mala: dual averaging on the acceptance rate (Hoffman & Gelman
        2014). sgld: legacy Robbins-Monro on the drift/noise ratio.
    adapt_target : float or None
        Target acceptance rate (hmc: 0.8, mala: 0.574) or drift/noise ratio
        (sgld: 0.3). None picks the kernel default.
    traj_length : float or None
        hmc: target trajectory length ``eps·L`` in preconditioned-σ units
        (default 1.5 ≈ π/2, the decorrelation scale of a unit Gaussian).
        Each iteration draws L uniformly in [1, clip(ceil(traj_length/eps),
        1, n_leapfrog)] — when curvature forces eps down, L rises to
        compensate, so proposals keep travelling O(1) posterior widths.
        Ignored for mala (L=1) and sgld.
    n_leapfrog : int
        hmc: hard cap on leapfrog steps per iteration (cost/memory guard on
        the traj_length rule above).
    precond : {"diag_hessian", "full_hessian", "rmsprop", "identity"}
        Mass-matrix preconditioner, built once at the init point.
        - ``diag_hessian`` (recommended default): inverse |diag(H)|.
        - ``full_hessian``: inverse |H| via eigendecomposition; captures
          cross-parameter correlations (useful when parameters trade off,
          e.g. shape↔temperature degeneracies). O(D^3) once.
        - ``rmsprop`` (sgld only): running EMA of grad², frozen at burn-in end.
        - ``identity``: no preconditioning (diagnostics only).
    polish_map : bool
        If True, run a brief LBFGS on the Jacobian-corrected target to
        recenter chains on the posterior mode. Only meaningful when the
        Jacobian shift is non-negligible (i.e. ``temperature ≈ 1``).
    chunk_size : int
        Iterations per jitted `lax.scan` chunk (progress-bar granularity and
        compile-size bound; does not change the math).
    laplace_tol : float
        Relative eigenvalue threshold flagging non-identified directions in
        the Laplace covariance.

    Returns
    -------
    SimpleNamespace — samples, summary, diagnostics, Laplace covariance and
    metadata (see the fields set at the end of this function).
    """
    if kernel not in _KERNELS:
        raise ValueError(f"Unknown kernel: {kernel!r}. Choose from {_KERNELS}.")
    if burn_in is None:
        burn_in = n_samples
    if adapt_target is None:
        adapt_target = {"hmc": 0.8, "mala": 0.574, "sgld": 0.3}[kernel]
    elif kernel in ("hmc", "mala") and adapt_target < 0.4:
        # Old SGLD decks carry adapt_target ≈ 0.3 (a drift/noise ratio); for
        # the Metropolis kernels the same key is an acceptance-rate target,
        # where 0.3 is a poor choice. Likely a stale deck setting.
        print(f"  [sampling] WARNING: adapt_target={adapt_target} with "
              f"kernel={kernel!r} targets a {adapt_target:.0%} acceptance "
              f"rate — this looks like an SGLD-era drift/noise setting. "
              f"Remove adapt_target from the deck to use the {kernel} "
              f"default ({0.8 if kernel == 'hmc' else 0.574}).")
    if step_size is None:
        step_size = 0.1 if kernel == "sgld" else 0.5
    if kernel == "mala":
        n_leapfrog = 1
    n_leapfrog = max(1, int(n_leapfrog))
    if traj_length is None:
        traj_length = 1.5
    traj_length = float(traj_length)
    if kernel in ("hmc", "mala") and precond == "rmsprop":
        raise ValueError(
            "precond='rmsprop' is only supported with kernel='sgld' "
            "(the Metropolis kernels need a fixed mass matrix). "
            "Use 'diag_hessian' or 'full_hessian'."
        )

    problem_s = build_sampling_problem(problem, temperature=temperature)
    # Convert the fit-space (arcsin) MAP into the sampler's coordinates
    # (logistic for two-sided bounds — see _make_sampling_bijector).
    u_map = jnp.asarray(problem_s.u_from_fit(u_map), dtype=jnp.float64)
    D = int(u_map.shape[0])

    # Optionally polish to the Jacobian-corrected mode.
    if polish_map:
        u_chain_init = _polish_map(problem_s, u_map, max_iter=polish_max_iter)
    else:
        u_chain_init = u_map

    # ─── Hessian (once) → preconditioner + Laplace covariance ────────────
    # ``rmsprop`` is adaptive: M_diag is updated during burn-in from an EMA
    # of (mean across chains of) grad^2, then frozen for sampling so the
    # sampler targets a fixed invariant distribution.
    kind = "diag" if precond in ("diag_hessian", "identity", "rmsprop") else "full"
    hessian_u = None
    if precond in ("diag_hessian", "full_hessian"):
        hessian_u = _full_hessian(problem_s, u_map)

    rmsprop = precond == "rmsprop"
    if precond == "diag_hessian":
        M_diag = _build_diag_hessian_precond(problem_s, u_chain_init,
                                             H=(hessian_u if not polish_map
                                                else None))
        precond_obj = M_diag
    elif precond == "identity":
        M_diag = jnp.ones(D, dtype=jnp.float64)
        precond_obj = M_diag
    elif precond == "rmsprop":
        M_diag = jnp.ones(D, dtype=jnp.float64)
        precond_obj = M_diag
    elif precond == "full_hessian":
        precond_obj = _build_full_hessian_precond(
            problem_s, u_chain_init,
            H=(hessian_u if not polish_map else None))
    else:
        raise ValueError(f"Unknown precond: {precond!r}. Choose "
                         "'diag_hessian', 'full_hessian', 'rmsprop', or 'identity'.")

    # Laplace physical-parameter covariance at the MAP (delta method through
    # the constraint map). Σ_phys is singular along the constraints, but its
    # diagonal gives valid 1σ physical error bars (exported as laplace/sigma),
    # and its flat directions diagnose degeneracies.
    lap = None
    if hessian_u is not None:
        lap = _laplace_physical_cov(problem_s, u_map, hessian_u,
                                    tol=laplace_tol)
        _print_nonidentified(lap)

    # ─── backends: chains across devices (pmap) or one device (vmap) ─────
    n_dev = _jax.device_count()
    use_chain_pmap = (not serial_requested()) and n_dev > 1 and n_chains <= n_dev
    chain_devices = _jax.devices()[:n_chains] if use_chain_pmap else None

    # Initialize chain states: perturb in sigma_u units (sqrt of diag precond)
    # so perturb_scale=1.0 means "start chains roughly 1 posterior std apart
    # from the MAP". This is appropriate for R-hat sensitivity.
    key = jr.PRNGKey(seed)
    key, sk = jr.split(key)
    if kind == "diag":
        sigma_u = jnp.sqrt(precond_obj)
    else:
        sigma_u = jnp.sqrt(jnp.diag(precond_obj[0]))
    perturb = perturb_scale * sigma_u[None, :] * jr.normal(
        sk, (n_chains, D), dtype=jnp.float64
    )
    us = u_chain_init[None, :] + perturb                              # (n_chains, D)

    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(desc=f"{kernel.upper()} burn-in",
                   total=burn_in + n_samples * thin, unit="iter")

    t0 = time.time()
    eps_history = np.zeros(burn_in, dtype=np.float64)
    n_divergent_burn = 0

    avg_leapfrog = 0.0
    if kernel in ("hmc", "mala"):
        mu_da = float(np.log(10.0 * step_size))
        burn_chunk, samp_chunk = _make_hmc_chunk_fns(
            problem_s, kind, precond_obj, n_chains, mu_da, adapt_target,
            traj_length, n_leapfrog, use_chain_pmap, chain_devices)

        # Initial (logp, grad) at the chain states — carried thereafter, so
        # each HMC iteration costs exactly n_leap gradient evaluations.
        init_lp = jit(vmap(problem_s.target_log_prob_and_grad))
        logps, gs = init_lp(us)

        if not adapt_step:
            # Freeze eps by pinning the DA state to log(step_size).
            da = (jnp.zeros(()), jnp.zeros(()),
                  jnp.log(jnp.asarray(step_size)), jnp.log(jnp.asarray(step_size)))
        else:
            da = _da_init(step_size)

        # ── burn-in ──
        off = 0
        nl_sum = 0.0
        for size in _split_chunks(burn_in, chunk_size):
            key, keys = _chunk_keys(key, size, n_chains)
            key, lkeys = _chunk_lkeys(key, size)
            if adapt_step:
                us, logps, gs, da, (eps_h, alpha_h, div_h, nl_h) = burn_chunk(
                    us, logps, gs, da, keys, lkeys)
            else:
                eps_fix = jnp.asarray(step_size)
                us, logps, gs, (u_h, lp_h, acc_h, div_h, nl_h) = samp_chunk(
                    us, logps, gs, eps_fix, keys, lkeys)
                eps_h = jnp.full(size, step_size)
                alpha_h = jnp.mean(acc_h, axis=1)
            eps_history[off:off + size] = np.asarray(eps_h)
            n_divergent_burn += int(np.sum(np.asarray(div_h)))
            nl_sum += float(np.sum(np.asarray(nl_h)))
            off += size
            if bar is not None:
                bar.update(size)
                bar.set_postfix({"step_size": f"{float(eps_h[-1]):.2e}",
                                 "accept": f"{float(alpha_h[-1]):.2f}",
                                 "L": f"{float(nl_h[-1]):.0f}"})

        eps = float(np.exp(np.asarray(da[3]))) if adapt_step else float(step_size)

        if bar is not None:
            bar.set_description(f"{kernel.upper()} sampling")

        # ── sampling (fixed step size) ──
        n_iter_sample = n_samples * thin
        u_samples = np.empty((n_chains, n_samples, D), dtype=np.float64)
        log_probs = np.empty((n_chains, n_samples), dtype=np.float64)
        acc_all = []
        n_divergent_samp = 0
        keep_idx, it_global = 0, 0
        eps_j = jnp.asarray(eps)
        for size in _split_chunks(n_iter_sample, chunk_size):
            key, keys = _chunk_keys(key, size, n_chains)
            key, lkeys = _chunk_lkeys(key, size)
            us, logps, gs, (u_h, lp_h, acc_h, div_h, nl_h) = samp_chunk(
                us, logps, gs, eps_j, keys, lkeys)
            u_h, lp_h = np.asarray(u_h), np.asarray(lp_h)      # (size, nc, ...)
            acc_all.append(np.asarray(acc_h))
            n_divergent_samp += int(np.sum(np.asarray(div_h)))
            nl_sum += float(np.sum(np.asarray(nl_h)))
            # keep iterations where (global_it + 1) % thin == 0
            its = np.arange(it_global, it_global + size)
            sel = np.nonzero((its + 1) % thin == 0)[0]
            nk = len(sel)
            u_samples[:, keep_idx:keep_idx + nk] = np.moveaxis(u_h[sel], 0, 1)
            log_probs[:, keep_idx:keep_idx + nk] = np.moveaxis(lp_h[sel], 0, 1)
            keep_idx += nk
            it_global += size
            if bar is not None:
                bar.update(size)
        accept_rate = (np.concatenate(acc_all, axis=0).mean(axis=0)
                       if acc_all else np.full(n_chains, np.nan))
        n_divergent = n_divergent_burn + n_divergent_samp
        avg_leapfrog = nl_sum / max(1, burn_in + n_iter_sample)

    else:  # ── legacy SGLD ──
        burn_chunk, samp_chunk = _make_sgld_chunk_fns(
            problem_s, kind, precond_obj, n_chains, burn_in,
            adapt_step, adapt_target, rmsprop, use_chain_pmap, chain_devices)

        eps_j = jnp.asarray(float(step_size))
        rms = jnp.ones(D, dtype=jnp.float64)
        it0 = jnp.zeros(())
        off = 0
        for size in _split_chunks(burn_in, chunk_size):
            key, keys = _chunk_keys(key, size, n_chains)
            us, eps_j, rms, it0, eps_h = burn_chunk(us, eps_j, rms, it0, keys)
            eps_history[off:off + size] = np.asarray(eps_h)
            off += size
            if bar is not None:
                bar.update(size)
                bar.set_postfix({"step_size": f"{float(eps_j):.2e}"})

        eps = float(eps_j)
        # Frozen preconditioner for the sampling phase (rmsprop freezes here).
        M_frozen = (1.0 / (jnp.sqrt(rms) + 1e-6)) if rmsprop else \
            (precond_obj if kind == "diag" else jnp.zeros(()))

        if bar is not None:
            bar.set_description("SGLD sampling")

        n_iter_sample = n_samples * thin
        u_samples = np.empty((n_chains, n_samples, D), dtype=np.float64)
        log_probs = np.empty((n_chains, n_samples), dtype=np.float64)
        keep_idx, it_global = 0, 0
        for size in _split_chunks(n_iter_sample, chunk_size):
            key, keys = _chunk_keys(key, size, n_chains)
            us, (u_h, lp_h) = samp_chunk(us, eps_j, M_frozen, keys)
            u_h, lp_h = np.asarray(u_h), np.asarray(lp_h)
            its = np.arange(it_global, it_global + size)
            sel = np.nonzero((its + 1) % thin == 0)[0]
            nk = len(sel)
            u_samples[:, keep_idx:keep_idx + nk] = np.moveaxis(u_h[sel], 0, 1)
            log_probs[:, keep_idx:keep_idx + nk] = np.moveaxis(lp_h[sel], 0, 1)
            keep_idx += nk
            it_global += size
            if bar is not None:
                bar.update(size)
        accept_rate = np.full(n_chains, np.nan)   # unadjusted kernel
        n_divergent = 0

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
        method="mcmc",
        kernel=kernel,
        samples_phys=samples_phys,
        u_samples=u_samples,
        log_probs=log_probs,
        summary=summary,
        rhat=rhat_dict,
        ess=ess_dict,
        step_size_history=eps_history,
        u_chain_init=np.asarray(u_chain_init),
        u_map=np.asarray(u_map),
        hessian_u=(np.asarray(hessian_u) if hessian_u is not None else None),
        hessian_ref=np.asarray(u_map),
        cov_phys=(lap.cov_phys if lap is not None else None),
        cov_phys_labels=(lap.labels if lap is not None else None),
        laplace_sigma=(lap.sigma_by_prefix if lap is not None else None),
        n_nonidentified=(lap.n_nonidentified if lap is not None else 0),
        nonid_loadings=(lap.nonid_loadings if lap is not None else None),
        nonid_descriptions=(lap.nonid_descriptions if lap is not None else []),
        varying_keys=problem.varying_keys,
        prefixes=list(samples_phys),
        temperature=problem_s.temperature,
        n_pixels_valid=problem_s.n_pixels_valid,
        step_size_final=eps,
        accept_rate=accept_rate,
        n_divergent=n_divergent,
        n_leapfrog=(n_leapfrog if kernel in ("hmc", "mala") else 0),
        traj_length=(traj_length if kernel == "hmc" else 0.0),
        avg_leapfrog=avg_leapfrog,
        adapt_target=adapt_target,
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


def run_sgld_posterior(problem, u_map, **kwargs):
    """Backward-compatible alias: :func:`run_mcmc_posterior` with the legacy
    SGLD kernel unless the caller picks another one."""
    kwargs.setdefault("kernel", "sgld")
    return run_mcmc_posterior(problem, u_map, **kwargs)


def run_laplace_posterior(problem, u_map, *,
                          temperature=None,
                          polish_map=False, polish_max_iter=200,
                          laplace_tol=1e-8):
    """Hessian-only (Laplace) error bars at the MAP — no chains.

    Computes the full Hessian of the Jacobian-corrected log-posterior at
    ``u_map`` (analytic, or batched finite differences on the general
    quadrature path), then the delta-method physical covariance
    ``Σ_phys = J Σ_u Jᵀ``. Seconds instead of minutes; exact if the posterior
    is Gaussian near the MAP, and a good first look before committing to a
    full MCMC run. Non-identified (flat) directions are dropped from Σ and
    reported with their physical-parameter loadings.

    Returns a SimpleNamespace field-compatible with the MCMC result
    (``summary`` has mean/std/p16/p50/p84/corr_intra; the sample- and
    chain-specific fields are ``None``).
    """
    problem_s = build_sampling_problem(problem, temperature=temperature)
    # Fit-space (arcsin) MAP → sampler coordinates (logistic for two-sided).
    u_map = jnp.asarray(problem_s.u_from_fit(u_map), dtype=jnp.float64)

    if polish_map:
        u_ref = _polish_map(problem_s, u_map, max_iter=polish_max_iter)
    else:
        u_ref = u_map

    t0 = time.time()
    hessian_u = _full_hessian(problem_s, u_ref)
    lap = _laplace_physical_cov(problem_s, u_ref, hessian_u, tol=laplace_tol)
    _print_nonidentified(lap)

    d0 = {k: np.asarray(v) for k, v in problem_s.resolve_one(u_ref).items()}
    labels = lap.labels
    Sig = lap.cov_phys
    sig_flat = np.sqrt(np.clip(np.diag(Sig), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_full = Sig / np.outer(sig_flat, sig_flat)
    corr_full = np.where(np.isfinite(corr_full), corr_full, 0.0)
    np.fill_diagonal(corr_full, 1.0)

    summary, i = {}, 0
    for prefix, x in d0.items():
        nt = x.shape[0]
        sig = lap.sigma_by_prefix[prefix]
        summary[prefix] = {
            "mean": x,
            "std":  sig,
            "p16":  x - sig,
            "p50":  x.copy(),
            "p84":  x + sig,
            "corr_intra": corr_full[i:i + nt, i:i + nt],
        }
        i += nt

    t1 = time.time()
    return SimpleNamespace(
        method="laplace",
        kernel="laplace",
        samples_phys=None,
        u_samples=None,
        log_probs=None,
        summary=summary,
        rhat=None,
        ess=None,
        step_size_history=None,
        u_chain_init=np.asarray(u_ref),
        u_map=np.asarray(u_map),
        hessian_u=np.asarray(hessian_u),
        hessian_ref=np.asarray(u_ref),
        cov_phys=lap.cov_phys,
        cov_phys_labels=lap.labels,
        laplace_sigma=lap.sigma_by_prefix,
        n_nonidentified=lap.n_nonidentified,
        nonid_loadings=lap.nonid_loadings,
        nonid_descriptions=lap.nonid_descriptions,
        varying_keys=problem.varying_keys,
        prefixes=list(summary),
        temperature=problem_s.temperature,
        n_pixels_valid=problem_s.n_pixels_valid,
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
