"""Differentiable Thomson-scattering fit on time-resolved streaks.

Public entry points:

- :class:`Param`      — tiny dataclass holding ``(value, min, max, vary)``.
- :func:`build_params` — assemble a ``{name: Param}`` dict for a given
  ``(Nelectrons, Nions, Nt)`` problem, with the three-level specificity
  override pattern used throughout the package.
- :func:`run_fit_grad` — run a JAX + optax fit on a time-resolved Thomson
  streak. Returns ``(result, best_fit)``.
- :func:`compute_initial_fit` — evaluate the forward model at the initial
  guess (no fitting). Useful for diagnostic plots.
"""
import math
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

# Force float64 BEFORE the first jax.numpy import on this module's path.
# Belt-and-suspenders to the same call in ThomsonScattering/__init__.py.
import jax as _jax
_jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad
from scipy.constants import k as kB, e

from .arrays import extract_params_as_array
from .forward import scattered_power_wavelength


# ─── parameter container ──────────────────────────────────────────────────────

@dataclass(slots=True)
class Param:
    """Per-time-step parameter spec consumed by the fitter."""
    value: float
    min: float = -math.inf
    max: float = math.inf
    vary: bool = True


# ─── Tikhonov penalty ────────────────────────────────────────────────────────

def _tikhonov_penalty(param_array, profile_axis, lambda_weights, thresholds,
                      relative=True, norm_scale=1, monotonic=0):
    if len(lambda_weights) != 3:
        raise ValueError(
            f"lambda_weights must have exactly 3 elements, got {len(lambda_weights)}"
        )

    penalty = 0

    if not hasattr(norm_scale, "__len__"):
        norm_scale = [norm_scale] * 3
    if not hasattr(monotonic, "__len__"):
        monotonic = [monotonic] * 3

    dt = jnp.diff(profile_axis)             # (N-1,)
    dt_mid = (dt[:-1] + dt[1:]) / 2         # (N-2,)
    d1 = jnp.diff(param_array) / dt         # (N-1,)
    d2 = jnp.diff(d1) / dt_mid              # (N-2,)
    derivs = [param_array, d1, d2]

    rf0 = 1 + relative * (jnp.abs(param_array) - 1)
    rf1 = (rf0[:-1] + rf0[1:]) / 2
    rf2 = (rf0[:-2] + rf0[1:-1] + rf0[2:]) / 3
    relative_factors = [rf0, rf1, rf2]

    for order in range(3):
        deriv = derivs[order]
        relative_factor = relative_factors[order]
        current_threshold = thresholds[order] / relative_factor
        current_norm = norm_scale[order] * relative_factor
        if monotonic[order] == 0:
            signed_deriv = jnp.abs(deriv)
        else:
            signed_deriv = monotonic[order] * deriv
        adjusted_deriv = jnp.maximum(0, signed_deriv - current_threshold) / current_norm
        penalty += lambda_weights[order] * jnp.mean(adjusted_deriv ** 2)

    return penalty


# ─── constraint compilation ──────────────────────────────────────────────────

# Namespace exposed to constraint expressions. `min` / `max` are the binary
# jnp variants — matches the 2-arg form documented in the deck schema.
_CONSTRAINT_NS = {
    "min": jnp.minimum, "max": jnp.maximum,
    "abs": jnp.abs, "where": jnp.where, "clip": jnp.clip,
    "sqrt": jnp.sqrt, "exp": jnp.exp, "log": jnp.log,
    "__builtins__": {},
}


def _compile_grad_constraints(constraints):
    """Convert {prefix: str|callable} into {prefix: callable(p) -> (Nt,) array}."""
    compiled = {}
    for prefix, spec in constraints.items():
        if callable(spec):
            compiled[prefix] = spec
        elif isinstance(spec, str):
            code = compile(spec, f"<constraint:{prefix}>", "eval")
            def _fn(p, _code=code, _prefix=prefix):
                try:
                    return eval(_code, _CONSTRAINT_NS, p)
                except NameError as nerr:
                    raise NameError(
                        f"Constraint for {_prefix!r} references unknown name "
                        f"({nerr}). Available names: {sorted(p)}"
                    ) from None
            compiled[prefix] = _fn
        else:
            raise TypeError(
                f"Constraint for {prefix!r} must be a str or callable, "
                f"got {type(spec).__name__}"
            )
    return compiled


# ─── parameter building ──────────────────────────────────────────────────────

def build_params(Nelectrons, Nions, Nt, params_settings=None, background_order=None):
    """Build a ``{name: Param}`` dict with the package's naming scheme.

    Parameters are keyed ``<var><species>_<time>`` (e.g. ``Te0_3``, ``Ti1_0``);
    the per-shot density ``n_<time>`` has no species index.

    ``params_settings`` is a ``dict[str, dict]`` mapping parameter keys to
    Param-constructor kwargs (``value``, ``min``, ``max``, ``vary``). Keys use
    three-level specificity:

    - per-time, per-species: ``"Te0_3"``  → species 0 at t=3 only
    - per-species:           ``"Te0"``    → all times for species 0
    - global:                ``"Te"``     → all Te species at all times

    For ``n`` (no species index) the lookup checks ``"n_<t>"`` then ``"n"``.

    If ``background_order`` is not None, polynomial background coefficients
    ``bg<i>_<t>`` for ``i in range(background_order + 1)`` are added.
    """
    if params_settings is None:
        params_settings = {}

    def _lookup(base, species, t, default):
        if species is None:
            key_specific = f"{base}_{t}"
            key_species = None
        else:
            key_specific = f"{base}{species}_{t}"
            key_species = f"{base}{species}"
        key_global = base

        if key_specific in params_settings:
            user = params_settings[key_specific]
        elif key_species is not None and key_species in params_settings:
            user = params_settings[key_species]
        elif key_global in params_settings:
            user = params_settings[key_global]
        else:
            user = {}
        return {**default, **user}

    p = {}

    for t in range(Nt):
        p[f"n_{t}"] = Param(**_lookup("n", None, t, {"value": 1e20}))

    for s in range(Nelectrons):
        for t in range(Nt):
            p[f"Te{s}_{t}"]     = Param(**_lookup("Te",     s, t, {"value": 100.0}))
            p[f"ue{s}_{t}"]     = Param(**_lookup("ue",     s, t, {"value": 0.0}))
            p[f"pe{s}_{t}"]     = Param(**_lookup("pe",     s, t, {"value": 2.0}))
            p[f"efract{s}_{t}"] = Param(**_lookup("efract", s, t, {"value": 1.0}))

    for s in range(Nions):
        for t in range(Nt):
            p[f"Ti{s}_{t}"]     = Param(**_lookup("Ti",     s, t, {"value": 100.0}))
            p[f"ui{s}_{t}"]     = Param(**_lookup("ui",     s, t, {"value": 0.0}))
            p[f"pi{s}_{t}"]     = Param(**_lookup("pi",     s, t, {"value": 2.0}))
            p[f"ifract{s}_{t}"] = Param(**_lookup("ifract", s, t, {"value": 1.0}))

    if background_order is not None:
        for i in range(background_order + 1):
            for t in range(Nt):
                p[f"bg{i}_{t}"] = Param(**_lookup("bg", i, t, {"value": 0.0}))

    return p


def _expand_extra_params(params, extra_params, Nt):
    """Inject extras as per-time replicated free vars and return their prefixes.

    Each entry in ``extra_params`` is a dict with a ``name`` key plus any
    Param kwargs (``value``, ``min``, ``max``, ``vary``). Array-valued
    fields are indexed per time step.
    """
    prefixes = []
    if extra_params is None:
        return prefixes
    for extra_def in extra_params:
        ed = dict(extra_def)
        name = ed.pop("name")
        prefixes.append(name)
        for t in range(Nt):
            kw = {
                k: float(v[t]) if (hasattr(v, "__len__") and not isinstance(v, str)) else v
                for k, v in ed.items()
            }
            params[f"{name}_{t}"] = Param(**kw)
    return prefixes


# ─── forward-model call (shared by run_fit_grad and compute_initial_fit) ─────

_jitted_scattered_power_wavelength = jit(
    scattered_power_wavelength,
    static_argnames=("normalization_type", "notch", "irf_normalization", "gain_mode"),
)


def _call_forward(n, Te, ue, pe, efract, Ti, ui, pi, ifract, bg, ms):
    """Invoke the JIT-wrapped forward model with package unit conventions.

    ``n`` is in cm^-3, ``Te`` / ``Ti`` in eV; converted to m^-3 and K inside.
    """
    return _jitted_scattered_power_wavelength(
        n=n * 1e6,
        ue=ue, ui=ui,
        Te=Te * e / kB, Ti=Ti * e / kB,
        pe=pe, pi=pi,
        efract=efract, ifract=ifract,
        ion_z=ms["ion_z"],
        ion_a=ms["ion_a"],
        wavelengths=ms["wavelengths"],
        probe_wavelength=ms["probe_wavelength"],
        probe_vec=ms["probe_vec"],
        scatter_vec=ms["scatter_vec"],
        ue_dir=ms["ue_dir"],
        ui_dir=ms["ui_dir"],
        instr_func_arr=ms.get("instr_func_arr", None),
        irf_normalization=ms.get("irf_normalization", "area"),
        throughput=ms.get("throughput", None),
        aperture_weights=ms.get("aperture_weights", None),
        background_coefs=bg,
        normalization_type=ms.get("normalization_type", "max"),
        normalization_scale=ms.get("normalization_scale", 1),
        notch=ms.get("notch", None),
        probe_intensity=ms.get("probe_intensity", 0.0),
        probe_diameter=ms.get("probe_diameter", 1.0),
        pol_p_fraction=ms.get("pol_p_fraction", 1.0),
        gain_mode=ms.get("gain_mode", "off"),
    )


# ─── data-fidelity term ─────────────────────────────────────────────────────

def _log_likelihood(fit, data, variance):
    # Replace NaN (notch pixels) *before* arithmetic so the VJP never sees
    # 0 * NaN = NaN. fit and data are set equal (residual = 0); var = 1 avoids /0.
    mask   = jnp.isnan(data) | jnp.isnan(fit)
    fit_s  = jnp.where(mask, 0.0, fit)
    data_s = jnp.where(mask, 0.0, data)
    var_s  = jnp.where(mask, 1.0, variance)
    r = (fit_s - data_s) ** 2 / var_s
    return jnp.sum(r) / jnp.sum(~mask)


# ─── grad-problem assembly ──────────────────────────────────────────────────

def _build_penalty_list(penalty_settings, Nelectrons, Nions, background_order):
    """Return ``list[(base, species_or_None, pset)]`` matched against prefixes.

    Most-specific match wins (``Te0`` before ``Te``); ``n`` has no species idx.
    The for-loop is unrolled by jit at trace time, so penalty selection is free.
    """
    if penalty_settings is None:
        return []
    bg_block = ([("bg", background_order + 1)]
                if background_order is not None else [])
    out = []
    for base, n_sp in ([("n", 1)] + bg_block
                       + [(b, Nelectrons) for b in ("Te", "ue", "pe", "efract")]
                       + [(b, Nions)      for b in ("Ti", "ui", "pi", "ifract")]):
        for s in range(n_sp):
            prefix = base if base == "n" else f"{base}{s}"
            pset = penalty_settings.get(prefix)
            if pset is None:
                pset = penalty_settings.get(base)
            if pset is not None:
                out.append((base, None if base == "n" else s, pset))
    return out


def _make_unconstrained_transforms(lower_np, upper_np):
    """Build the lmfit-style arcsin/sqrt/identity transform pair.

    The optimizer sees a problem where every coordinate is O(1) regardless of
    physical units. Bounded → arcsin; one-sided → sqrt; unbounded → identity.
    """
    lower = jnp.array(lower_np)
    upper = jnp.array(upper_np)

    def to_internal_np(x, lo, hi):
        if np.isfinite(lo) and np.isfinite(hi):
            return np.arcsin(np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0))
        elif np.isfinite(lo):
            return np.sqrt(max((x - lo + 1.0) ** 2 - 1.0, 0.0))
        elif np.isfinite(hi):
            return np.sqrt(max((hi - x + 1.0) ** 2 - 1.0, 0.0))
        else:
            return x

    def to_external_jax(u):
        lo_fin  = jnp.isfinite(lower)
        hi_fin  = jnp.isfinite(upper)
        bounded = lo_fin & hi_fin
        lo_only = lo_fin & ~hi_fin
        hi_only = ~lo_fin & hi_fin
        x_b = lower + (upper - lower) / 2.0 * (jnp.sin(u) + 1.0)
        x_l = lower - 1.0 + jnp.sqrt(u ** 2 + 1.0)
        x_h = upper + 1.0 - jnp.sqrt(u ** 2 + 1.0)
        return jnp.where(bounded, x_b,
                         jnp.where(lo_only, x_l,
                                   jnp.where(hi_only, x_h, u)))

    return to_internal_np, to_external_jax, lower, upper


def _build_grad_problem(Pkl_data, Pkl_var, measurement_settings,
                        penalty_settings=None, params_settings=None,
                        constraints=None, extra_params=None):
    """Set up the differentiable Thomson-fit problem in unconstrained space.

    Returns a SimpleNamespace bundling every closure and metadatum that
    ``run_fit_grad`` (and future posterior samplers) need.
    """
    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    Nt = jnp.shape(Pkl_data)[1]
    background_order = measurement_settings.get("background_order", None)

    Pkl_data = jnp.array(Pkl_data)
    Pkl_var = jnp.array(Pkl_var)

    params = build_params(Nelectrons, Nions, Nt, params_settings,
                          background_order=background_order)
    _extra_prefixes = _expand_extra_params(params, extra_params, Nt)
    _constraints = _compile_grad_constraints(constraints) if constraints else {}

    def _prefix_of(key):
        return key.rsplit("_", 1)[0]

    constrained_prefixes = set(_constraints)

    varying_keys = [
        k for k, p in params.items()
        if p.vary and _prefix_of(k) not in constrained_prefixes
    ]
    fixed_vals = {
        k: float(p.value) for k, p in params.items()
        if not p.vary and _prefix_of(k) not in constrained_prefixes
    }
    key_to_idx = {k: i for i, k in enumerate(varying_keys)}

    x0_np    = np.array([params[k].value for k in varying_keys], dtype=np.float64)
    lower_np = np.array([params[k].min   for k in varying_keys], dtype=np.float64)
    upper_np = np.array([params[k].max   for k in varying_keys], dtype=np.float64)

    to_internal_np, to_external_jax, lower, upper = (
        _make_unconstrained_transforms(lower_np, upper_np)
    )

    u0 = np.array([to_internal_np(x0_np[i], lower_np[i], upper_np[i])
                   for i in range(len(x0_np))], dtype=np.float64)

    penalty_list = _build_penalty_list(
        penalty_settings, Nelectrons, Nions, background_order
    )

    def _get(x, key):
        if key in key_to_idx:
            return x[key_to_idx[key]]
        return jnp.array(fixed_vals[key])

    def _unpack(x):
        # p accumulates {prefix: (Nt,) array} so constraints can reference
        # previously assembled prefixes by name.
        p = {}

        # Extras first: constraints may reference them by their bare name.
        for extra_prefix in _extra_prefixes:
            p[extra_prefix] = jnp.stack(
                [_get(x, f"{extra_prefix}_{t}") for t in range(Nt)]
            )

        def _row(base, s):
            prefix = base if base == "n" else f"{base}{s}"
            if prefix in _constraints:
                arr = _constraints[prefix](p)
            else:
                arr = jnp.stack([_get(x, f"{prefix}_{t}") for t in range(Nt)])
            p[prefix] = arr
            return arr

        n      = _row("n", None)
        Te     = jnp.stack([_row("Te",     s) for s in range(Nelectrons)])
        ue     = jnp.stack([_row("ue",     s) for s in range(Nelectrons)])
        pe     = jnp.stack([_row("pe",     s) for s in range(Nelectrons)])
        efract = jnp.stack([_row("efract", s) for s in range(Nelectrons)])
        Ti     = jnp.stack([_row("Ti",     s) for s in range(Nions)])
        ui     = jnp.stack([_row("ui",     s) for s in range(Nions)])
        pi_arr = jnp.stack([_row("pi",     s) for s in range(Nions)])
        ifract = jnp.stack([_row("ifract", s) for s in range(Nions)])
        bg = (jnp.stack([_row("bg", i) for i in range(background_order + 1)])
              if background_order is not None else None)
        return n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg

    def _build_params_dict(x):
        """{prefix: (Nt,) array} for every parameter (free, fixed, constrained, extra)."""
        p = {}
        for ep in _extra_prefixes:
            p[ep] = jnp.stack([_get(x, f"{ep}_{t}") for t in range(Nt)])

        def _fill(base, s):
            prefix = base if base == "n" else f"{base}{s}"
            if prefix in _constraints:
                arr = _constraints[prefix](p)
            else:
                arr = jnp.stack([_get(x, f"{prefix}_{t}") for t in range(Nt)])
            p[prefix] = arr

        _fill("n", None)
        for s in range(Nelectrons):
            for b in ("Te", "ue", "pe", "efract"):
                _fill(b, s)
        for s in range(Nions):
            for b in ("Ti", "ui", "pi", "ifract"):
                _fill(b, s)
        if background_order is not None:
            for i in range(background_order + 1):
                _fill("bg", i)
        return p

    def _forward(n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg):
        return _call_forward(n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg,
                             measurement_settings)

    # objective_flat must NOT be jit-decorated: jit(value_and_grad(f)) works,
    # but value_and_grad(jit(f)) cannot differentiate through the jit boundary.
    def objective_flat(x):
        n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg = _unpack(x)
        fit = _forward(n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg)
        loss = _log_likelihood(fit, Pkl_data, Pkl_var)
        arr_map = {
            "n": n, "Te": Te, "ue": ue, "pe": pe, "efract": efract,
            "Ti": Ti, "ui": ui, "pi": pi_arr, "ifract": ifract,
        }
        if bg is not None:
            arr_map["bg"] = bg
        for base, species, pset in penalty_list:
            arr = arr_map[base] if species is None else arr_map[base][species]
            loss = loss + _tikhonov_penalty(arr, **pset)
        return loss

    def objective_u(u):
        return objective_flat(to_external_jax(u))

    val_and_grad_fn = jit(value_and_grad(objective_u))

    return SimpleNamespace(
        objective_flat=objective_flat,
        objective_u=objective_u,
        val_and_grad_fn=val_and_grad_fn,
        to_external_jax=to_external_jax,
        to_internal_np=to_internal_np,
        build_params_dict=_build_params_dict,
        forward=_forward,
        unpack=_unpack,
        varying_keys=varying_keys,
        x0_np=x0_np, u0=u0,
        lower_np=lower_np, upper_np=upper_np,
        lower=lower, upper=upper,
        Nt=Nt,
    )


# ─── optimizer machinery ─────────────────────────────────────────────────────

def _build_optimizer(name, lr, fit_settings):
    """Return ``(opt, needs_value_fn)`` for the named optax optimizer.

    ``fit_settings`` is forwarded as kwargs to the constructor; remove any
    keys that don't apply before calling.
    """
    if name == "lbfgs":
        return optax.lbfgs(**fit_settings), True
    if name == "adam":
        return optax.adam(learning_rate=lr, **fit_settings), False
    if name == "adamw":
        return optax.adamw(learning_rate=lr, **fit_settings), False
    raise ValueError(
        f"Unknown optimizer {name!r}. "
        "Use 'lbfgs', 'adam', 'adamw', or 'sgld_lbfgs'."
    )


def _make_step_fn(opt, val_and_grad_fn, objective_u, needs_value_fn):
    """JIT-compile one optax update step."""
    if needs_value_fn:
        @jit
        def step(x, state):
            val, grad = val_and_grad_fn(x)
            updates, new_state = opt.update(
                grad, state, x, value=val, grad=grad, value_fn=objective_u,
            )
            return optax.apply_updates(x, updates), new_state, val
    else:
        @jit
        def step(x, state):
            val, grad = val_and_grad_fn(x)
            updates, new_state = opt.update(grad, state, x)
            return optax.apply_updates(x, updates), new_state, val
    return step


def _run_loop(x, state, step_fn, max_iter, tol, bar, nit_offset=0):
    """Run an optimizer loop with 100-iter-window convergence check.

    Returns ``(x, state, val_f, nit, converged)``.
    """
    window = deque(maxlen=100)
    converged = False
    val_f = float("inf")
    nit = nit_offset
    for _ in range(max_iter):
        x, state, val = step_fn(x, state)
        nit += 1
        val_f = float(val)
        window.append(val_f)
        if bar is not None:
            bar.update(1)
            postfix = {"obj": f"{val_f:.4g}"}
            if len(window) == window.maxlen:
                rel = (window[0] - min(window)) / (abs(window[0]) + 1e-300)
                postfix["d_obj_rel"] = f"{rel:.2e}"
            bar.set_postfix(postfix)
        if len(window) == window.maxlen:
            rel = (window[0] - min(window)) / (abs(window[0]) + 1e-300)
            if rel < tol:
                converged = True
                break
    return x, state, val_f, nit, converged


def _run_sgld_phase(x, val_and_grad_fn, sgld_lr, noise_scale, noise_decay,
                    seed, n_iters, bar):
    """SGLD warmup — returns ``(best_x, best_val)`` over the trajectory."""
    opt_sgld = optax.chain(
        optax.add_noise(eta=noise_scale, gamma=noise_decay, seed=seed),
        optax.scale(-sgld_lr),
    )
    state = opt_sgld.init(x)
    best_x, best_val = x, float("inf")

    @jit
    def _sgld_step(x, state):
        val, grad = val_and_grad_fn(x)
        updates, new_state = opt_sgld.update(grad, state, x)
        return optax.apply_updates(x, updates), new_state, val

    for _ in range(n_iters):
        x, state, val = _sgld_step(x, state)
        v = float(val)
        if v < best_val:
            best_val, best_x = v, x
        if bar is not None:
            bar.update(1)
            bar.set_postfix({"obj": f"{v:.4g}"})
    return best_x, best_val


# ─── public fit entry point ─────────────────────────────────────────────────

def run_fit_grad(Pkl_data, Pkl_var, measurement_settings,
                 penalty_settings=None, params_settings=None,
                 constraints=None, extra_params=None,
                 fit_settings=None, progress=False):
    """Run the Thomson scattering fit using JAX autodiff + optax.

    Computes exact gradients via :func:`jax.value_and_grad` and steps with an
    optax optimizer. Defaults to ``optax.lbfgs`` (with built-in Zoom line
    search) — the right choice for Thomson fits where the loss is mostly
    quadratic near the minimum. Adam / AdamW are available for cases where
    LBFGS gets stuck (typically when ``pe`` / ``pi`` are free and
    ``gammaincc`` produces near-flat regions).

    Bounds are enforced by the unconstrained reparameterization (arcsin /
    sqrt / identity), so no clipping is needed inside the loop.

    Parameters
    ----------
    Pkl_data, Pkl_var, measurement_settings, penalty_settings, params_settings :
        See the package's deck schema and ``build_params``.
    constraints : dict[str, str|callable] or None
        Equality-style reparameterization. Keys are parameter prefixes (e.g.
        ``"ifract1"``); values are either string expressions (evaluated against
        the dict of accumulated ``(Nt,)`` arrays with ``min``, ``max``,
        ``abs``, ``where``, ``clip``, ``sqrt``, ``exp``, ``log`` available)
        or callables receiving that dict.
    extra_params : list of dict or None
        Free dummy parameters injected into the fit (e.g. for use inside
        constraint expressions). Each dict needs ``name`` plus Param kwargs.
    fit_settings : dict or None
        Optimizer settings. Recognized keys:

        - ``optimizer`` (str, default ``"lbfgs"``): ``"lbfgs"``, ``"adam"``,
          ``"adamw"``, or ``"sgld_lbfgs"`` (SGLD warmup followed by LBFGS).
        - ``lr`` / ``learning_rate`` (float, default 1e-2): step size for
          adam / adamw.
        - ``max_iter`` (int, default 1000): hard iteration cap.
        - ``tol`` (float, default 1e-8): converged when the relative loss
          improvement over a 100-iteration window falls below ``tol``.
        - SGLD-specific: ``sgld_iter``, ``sgld_lr``, ``sgld_noise_scale``,
          ``sgld_noise_decay``, ``sgld_seed``.

        Any remaining keys flow through to the optax optimizer constructor.
    progress : bool
        If True, display a tqdm progress bar.

    Returns
    -------
    result : types.SimpleNamespace
        ``x``, ``varying_keys``, ``params_dict`` (prefix → (Nt,) array),
        ``fun`` (final loss), ``nit``, ``success``.
    best_fit : jnp.ndarray, shape (Nk, Nt)
        Forward model evaluated at the best-fit parameters.
    """
    if fit_settings is None:
        fit_settings = {}
    fit_settings = dict(fit_settings)
    optimizer_name = fit_settings.pop("optimizer", "lbfgs")
    lr = fit_settings.pop("lr", fit_settings.pop("learning_rate", 1e-2))
    max_iter = fit_settings.pop("max_iter", 1000)
    tol = fit_settings.pop("tol", 1e-8)
    sgld_lr          = fit_settings.pop("sgld_lr",          1e-3)
    sgld_noise_scale = fit_settings.pop("sgld_noise_scale", 0.1)
    sgld_noise_decay = fit_settings.pop("sgld_noise_decay", 0.55)
    sgld_seed        = fit_settings.pop("sgld_seed",        0)
    sgld_iter        = fit_settings.pop("sgld_iter",        None)

    problem = _build_grad_problem(
        Pkl_data, Pkl_var, measurement_settings,
        penalty_settings=penalty_settings,
        params_settings=params_settings,
        constraints=constraints,
        extra_params=extra_params,
    )

    x = jnp.array(problem.u0)

    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(desc=f"run_fit_grad ({optimizer_name})",
                   unit="iter", total=max_iter)

    try:
        if optimizer_name == "sgld_lbfgs":
            sgld_budget = sgld_iter if sgld_iter is not None else max_iter // 2
            lbfgs_budget = max(0, max_iter - sgld_budget)
            x, _ = _run_sgld_phase(
                x, problem.val_and_grad_fn,
                sgld_lr, sgld_noise_scale, sgld_noise_decay, sgld_seed,
                sgld_budget, bar,
            )
            if bar is not None:
                bar.set_description("run_fit_grad (lbfgs)")
            opt, needs_value_fn = _build_optimizer("lbfgs", lr, fit_settings)
            step_fn = _make_step_fn(opt, problem.val_and_grad_fn,
                                    problem.objective_u, needs_value_fn)
            x, _, _, nit, converged = _run_loop(
                x, opt.init(x), step_fn, lbfgs_budget, tol, bar,
                nit_offset=sgld_budget,
            )
        else:
            opt, needs_value_fn = _build_optimizer(optimizer_name, lr, fit_settings)
            step_fn = _make_step_fn(opt, problem.val_and_grad_fn,
                                    problem.objective_u, needs_value_fn)
            x, _, _, nit, converged = _run_loop(
                x, opt.init(x), step_fn, max_iter, tol, bar,
            )
    finally:
        if bar is not None:
            bar.close()

    # Final loss at the final x (the val captured inside the loop is the
    # loss *before* the last update step).
    final_val, _ = problem.val_and_grad_fn(x)
    x_phys = np.array(problem.to_external_jax(x))
    result = SimpleNamespace(
        x=x_phys,
        fun=float(final_val),
        nit=nit,
        success=converged,
        varying_keys=problem.varying_keys,
    )
    final_x = jnp.array(result.x)
    result.params_dict = {k: np.array(v)
                          for k, v in problem.build_params_dict(final_x).items()}
    best_fit = problem.forward(*problem.unpack(final_x))
    return result, best_fit


# ─── initial-guess diagnostic ────────────────────────────────────────────────

def compute_initial_fit(measurement_settings, params_settings, extra_params, Nt):
    """Evaluate the forward model at the initial guess (no fitting).

    Constraints are NOT applied — the resulting spectrum reflects the
    parameter values as typed in the deck. Use for diagnostic plots before
    launching :func:`run_fit_grad`.
    """
    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    background_order = measurement_settings.get("background_order", None)

    params = build_params(Nelectrons, Nions, Nt, params_settings,
                          background_order=background_order)
    _expand_extra_params(params, extra_params, Nt)

    n      = extract_params_as_array(params, "n", Nt)
    Te     = jnp.stack([extract_params_as_array(params, f"Te{i}", Nt)
                        for i in range(Nelectrons)])
    ue     = jnp.stack([extract_params_as_array(params, f"ue{i}", Nt)
                        for i in range(Nelectrons)])
    pe     = jnp.stack([extract_params_as_array(params, f"pe{i}", Nt)
                        for i in range(Nelectrons)])
    efract = jnp.stack([extract_params_as_array(params, f"efract{i}", Nt)
                        for i in range(Nelectrons)])
    Ti     = jnp.stack([extract_params_as_array(params, f"Ti{i}", Nt)
                        for i in range(Nions)])
    ui     = jnp.stack([extract_params_as_array(params, f"ui{i}", Nt)
                        for i in range(Nions)])
    pi_arr = jnp.stack([extract_params_as_array(params, f"pi{i}", Nt)
                        for i in range(Nions)])
    ifract = jnp.stack([extract_params_as_array(params, f"ifract{i}", Nt)
                        for i in range(Nions)])
    bg = None
    if background_order is not None:
        bg = jnp.stack([extract_params_as_array(params, f"bg{i}", Nt)
                        for i in range(background_order + 1)])

    return _call_forward(n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg,
                         measurement_settings)
