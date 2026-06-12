"""L-curve sweep for Tikhonov regularization in Thomson-scattering fits.

Repeatedly calls :func:`ThomsonScatteringArbitrary.fitting.run_fit_grad` with the
penalty ``lambda_weights`` scaled by each entry of ``lambda_scale``. The
*shape* of the regularizer (relative weights between prefixes and derivative
orders, plus ``norm_scale`` / ``relative`` / ``thresholds`` / ``monotonic`` /
``profile_axis``) is held fixed; only the global scalar moves.

Returns the L-curve (data residual vs. regularizer norm), per-fit results,
and the index of maximum curvature on the log-log curve — the canonical
"corner" pick for the optimal trade-off.

Warm-starting: when ``warm_start=True`` (default), an unregularized fit
(``lambda_weights = [0, 0, 0]``) is run first and its parameter values are
injected as the initial guess for every λ in the sweep. This is the strategy
that has empirically worked best on these problems; it avoids path-dependence
between sweep points.
"""
from __future__ import annotations

import re as _re
from types import SimpleNamespace

import numpy as np

from .fitting import _log_likelihood, _tikhonov_penalty, run_fit_grad
from .parallel import default_n_workers, parallel_map


_TRAILING_DIGITS_RE = _re.compile(r"^(.*?)(\d+)$")


def _split_prefix(prefix):
    """Return ``(base, species_str_or_None)`` for a parameter prefix.

    ``"Te0"`` → ``("Te", "0")``, ``"n"`` → ``("n", None)``,
    ``"bg2"`` → ``("bg", "2")``.
    """
    if prefix == "n":
        return "n", None
    m = _TRAILING_DIGITS_RE.match(prefix)
    if m:
        return m.group(1), m.group(2)
    return prefix, None


def _scale_penalty_settings(base_penalty_settings, scale):
    """Return a deep-enough copy with every ``lambda_weights`` multiplied by ``scale``."""
    if not base_penalty_settings:
        return base_penalty_settings
    out = {}
    for prefix, ps in base_penalty_settings.items():
        new_ps = dict(ps)
        lw = ps.get("lambda_weights")
        if lw is not None:
            new_ps["lambda_weights"] = [float(scale) * float(w) for w in lw]
        out[prefix] = new_ps
    return out


def _seed_warm_start(params_settings, extra_params, params_dict):
    """Return ``(new_params_settings, new_extra_params)`` with per-time values
    overridden by the (Nt,) arrays in ``params_dict``.

    Per-time-per-species keys (``"Te0_3"``) are the most specific tier in the
    ``build_params`` lookup, so writing them guarantees the warm-start values
    win regardless of what else lives in the dict. Existing min/max/vary
    fields are preserved by inheriting from the most-specific entry that
    already covers each time step.
    """
    extras_names = {e["name"] for e in extra_params} if extra_params else set()

    new_params = {k: dict(v) for k, v in (params_settings or {}).items()}

    for prefix, arr in params_dict.items():
        if prefix in extras_names:
            continue  # handled in the extras pass below
        arr = np.asarray(arr)
        Nt = len(arr)
        base, _ = _split_prefix(prefix)
        species_entry = new_params.get(prefix, {}) if prefix != base else {}
        global_entry  = new_params.get(base, {}) if base != prefix else {}
        for t in range(Nt):
            key = f"{prefix}_{t}"
            existing = new_params.get(key, {})
            merged = {**global_entry, **species_entry, **existing,
                      "value": float(arr[t])}
            new_params[key] = merged

    new_extras = None
    if extra_params:
        new_extras = []
        for entry in extra_params:
            ne = dict(entry)
            name = ne["name"]
            if name in params_dict:
                ne["value"] = np.asarray(params_dict[name])
            new_extras.append(ne)
    return new_params, new_extras


def _data_residual(best_fit, Pkl_data, Pkl_var):
    """Pure data-fidelity term — same `_log_likelihood` the optimizer uses."""
    import jax.numpy as jnp
    return float(_log_likelihood(jnp.asarray(best_fit),
                                 jnp.asarray(Pkl_data),
                                 jnp.asarray(Pkl_var)))


def _penalty_norm(params_dict, base_penalty_settings):
    """Sum of base-lambda-weighted Tikhonov terms — ``R(x)`` at ``scale=1``.

    This is the L-curve y-axis: the value of the regularizer functional
    evaluated at the fit's parameters, with the deck's base lambdas as fixed
    inter-term weights (so the relative importance of, say, Te0 vs. Ti0 vs.
    L1 vs. L2 is preserved as the user wrote it).
    """
    if not base_penalty_settings:
        return 0.0
    import jax.numpy as jnp
    total = 0.0
    for prefix, ps in base_penalty_settings.items():
        if prefix not in params_dict:
            continue
        arr = jnp.asarray(params_dict[prefix])
        total += float(_tikhonov_penalty(arr, **ps))
    return total


def _compute_curvature(residual_norm, penalty_norm):
    """Discrete log-log curvature; returns NaN at endpoints.

    Standard parametric-curve formula
    ``κ = (x' y'' - y' x'') / (x'^2 + y'^2)^{3/2}`` with x = log10(residual),
    y = log10(penalty). Derivatives via ``np.gradient`` (central differences,
    one-sided at the ends).
    """
    r = np.asarray(residual_norm, dtype=float)
    p = np.asarray(penalty_norm, dtype=float)
    if len(r) < 3:
        return np.full_like(r, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(np.where(r > 0, r, np.nan))
        y = np.log10(np.where(p > 0, p, np.nan))
    dx  = np.gradient(x)
    dy  = np.gradient(y)
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    denom = (dx ** 2 + dy ** 2) ** 1.5
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (dx * d2y - dy * d2x) / denom
    kappa[0] = np.nan
    kappa[-1] = np.nan
    return kappa


def _fit_one_sweep_point(task, progress=False):
    """Run one L-curve sweep point and return its picklable summary.

    Module-level (not a closure) so it can be shipped to a spawned worker by
    :func:`ThomsonScatteringArbitrary.parallel.parallel_map`. Returns everything the
    parent needs to assemble the L-curve, so no JAX objects cross the process
    boundary except plain numpy arrays.
    """
    (Pkl_data, Pkl_var, meas, scaled_pen, params_settings,
     constraints, extra_params, fit_settings, base_pen) = task
    result, best_fit = run_fit_grad(
        Pkl_data, Pkl_var, meas,
        penalty_settings=scaled_pen,
        params_settings=params_settings,
        constraints=constraints,
        extra_params=extra_params,
        fit_settings=fit_settings,
        progress=progress,
    )
    best_fit_np = np.asarray(best_fit)
    r = _data_residual(best_fit_np, Pkl_data, Pkl_var)
    p = _penalty_norm(result.params_dict, base_pen)
    return result, best_fit_np, r, p, float(result.fun)


def compute_L_curve(
    Pkl_data, Pkl_var, measurement_settings,
    penalty_settings,
    lambda_scale,
    *,
    params_settings=None,
    constraints=None,
    extra_params=None,
    fit_settings=None,
    warm_start=True,
    progress=True,
    n_workers=None,
):
    """Sweep a global ``lambda_scale`` multiplier over base penalty lambdas.

    Parameters
    ----------
    Pkl_data, Pkl_var, measurement_settings :
        Same shapes / semantics as :func:`run_fit_grad`.
    penalty_settings : dict[str, dict]
        Base penalty config; ``lambda_weights`` per prefix sets the *shape*
        of the regularizer (relative weights between prefixes and orders).
        Each fit multiplies these by one scalar from ``lambda_scale``.
    lambda_scale : array-like
        1-D positive scalars. ``compute_L_curve`` runs one fit per entry.
    params_settings, constraints, extra_params, fit_settings :
        Forwarded to :func:`run_fit_grad`. When ``warm_start=True`` the
        per-prefix initial values are overwritten by the unregularized fit's
        result before each sweep call; everything else (bounds, vary, etc.)
        is preserved.
    warm_start : bool, default True
        If True, run a single unregularized fit (all ``lambda_weights = 0``)
        first and warm-start every sweep fit from that solution. The unreg
        fit is also returned (``unreg_result`` / ``unreg_best_fit``).
    progress : bool, default True
        Show a tqdm bar per individual fit (forwarded to ``run_fit_grad``)
        and print a per-λ summary line as the sweep advances. Per-fit bars are
        suppressed when the sweep runs in parallel (they would interleave).
    n_workers : int or None, default None
        Number of parallel worker *processes* (independent fits running at once),
        NOT a core count — each worker runs one fit and itself uses ~3-4 cores.
        The sweep points are independent (all warm-start from the same
        unregularized fit), so they run across processes. ``None`` or ``1``
        (default) runs the sweep **sequentially**; ``N > 1`` runs N fits at a
        time; ``<= 0`` auto-sizes to the core budget (``cores // 4``, see
        :func:`ThomsonScatteringArbitrary.parallel.default_n_workers`). The warm-start fit
        always runs once, sequentially, first.

    Returns
    -------
    types.SimpleNamespace with fields:
        ``lambda_scale``       (N,) — original user order
        ``residual_norm``      (N,) — ``_log_likelihood`` at each fit
        ``penalty_norm``       (N,) — base-lambda-weighted R(x) at each fit
        ``curvature``          (N,) — log-log curvature (NaN at endpoints)
        ``optimal_index``      int  — argmax of curvature
        ``best_fits``          (N, Nk, Nt) — forward model per λ
        ``params``             dict[str, (N, Nt) ndarray]
        ``fit_results``        list[SimpleNamespace] from each run_fit_grad
        ``loss``               (N,) — raw final objective per fit
        ``optimal_result``, ``optimal_best_fit`` — convenience views
        ``unreg_result``, ``unreg_best_fit`` — None when ``warm_start=False``
    """
    lambda_scale = np.asarray(lambda_scale, dtype=float)
    if lambda_scale.ndim != 1 or len(lambda_scale) < 2:
        raise ValueError(
            f"lambda_scale must be a 1-D array of length >= 2, got shape "
            f"{lambda_scale.shape}"
        )
    if np.any(lambda_scale < 0):
        raise ValueError("lambda_scale entries must be non-negative.")

    unreg_result = None
    unreg_best_fit = None
    sweep_params_settings = params_settings
    sweep_extra_params    = extra_params

    if warm_start:
        print("L-curve: running unregularized warm-start fit "
              "(lambda_weights = [0, 0, 0])...")
        zero_pen = _scale_penalty_settings(penalty_settings, 0.0)
        unreg_result, unreg_best_fit_jax = run_fit_grad(
            Pkl_data, Pkl_var, measurement_settings,
            penalty_settings=zero_pen,
            params_settings=params_settings,
            constraints=constraints,
            extra_params=extra_params,
            fit_settings=fit_settings,
            progress=progress,
        )
        unreg_best_fit = np.asarray(unreg_best_fit_jax)
        print(f"  unreg fit: loss={unreg_result.fun:.4g}  "
              f"nit={unreg_result.nit}  success={unreg_result.success}")
        sweep_params_settings, sweep_extra_params = _seed_warm_start(
            params_settings, extra_params, unreg_result.params_dict,
        )

    fit_results   = []
    best_fits     = []
    residual_norm = np.empty(len(lambda_scale), dtype=float)
    penalty_norm  = np.empty(len(lambda_scale), dtype=float)
    loss_arr      = np.empty(len(lambda_scale), dtype=float)

    # One task per sweep point. All share the same warm-start seed and differ
    # only in the scaled penalty, so order is irrelevant — safe to fan out.
    tasks = [
        (Pkl_data, Pkl_var, measurement_settings,
         _scale_penalty_settings(penalty_settings, s),
         sweep_params_settings, constraints, sweep_extra_params,
         fit_settings, penalty_settings)
        for s in lambda_scale
    ]

    if n_workers is None or n_workers == 1:
        n_workers = 1                                   # default: sequential
    elif n_workers <= 0:
        n_workers = default_n_workers(len(lambda_scale))  # 0/negative => auto-size
    use_parallel = n_workers > 1 and len(lambda_scale) > 1

    outs = None
    if use_parallel:
        print(f"L-curve: fitting {len(lambda_scale)} lambda points across "
              f"{n_workers} worker processes...")
        try:
            outs = parallel_map(_fit_one_sweep_point, tasks, n_workers=n_workers)
        except Exception as exc:  # pickling / broken pool → serial fallback
            print(f"  parallel sweep failed ({exc!r}); running sequentially.")
            outs = None
    if outs is None:
        outs = []
        for i, t in enumerate(tasks):
            print(f"L-curve [{i + 1}/{len(lambda_scale)}]  "
                  f"lambda_scale = {lambda_scale[i]:.4g}")
            outs.append(_fit_one_sweep_point(t, progress=progress))

    for i, (result, best_fit_np, r, p, loss) in enumerate(outs):
        residual_norm[i] = r
        penalty_norm[i]  = p
        loss_arr[i]      = loss
        fit_results.append(result)
        best_fits.append(best_fit_np)
        print(f"  [{i + 1}/{len(lambda_scale)}]  lambda_scale={lambda_scale[i]:.4g} "
              f"→ residual={r:.4g}  penalty={p:.4g}  loss={loss:.4g}  "
              f"nit={result.nit}  success={result.success}")

    best_fits_arr = np.stack(best_fits, axis=0)

    # Stack params by prefix. All fits share the same set of prefixes (same
    # problem structure), so use the first fit's keys as the index.
    params_stack = {}
    for prefix in fit_results[0].params_dict:
        params_stack[prefix] = np.stack(
            [np.asarray(r.params_dict[prefix]) for r in fit_results], axis=0,
        )

    curvature = _compute_curvature(residual_norm, penalty_norm)
    if np.all(np.isnan(curvature)):
        optimal_index = int(np.argmin(loss_arr))  # fallback
    else:
        optimal_index = int(np.nanargmax(curvature))

    return SimpleNamespace(
        lambda_scale=lambda_scale,
        residual_norm=residual_norm,
        penalty_norm=penalty_norm,
        curvature=curvature,
        optimal_index=optimal_index,
        best_fits=best_fits_arr,
        params=params_stack,
        fit_results=fit_results,
        loss=loss_arr,
        optimal_result=fit_results[optimal_index],
        optimal_best_fit=best_fits_arr[optimal_index],
        unreg_result=unreg_result,
        unreg_best_fit=unreg_best_fit,
    )
