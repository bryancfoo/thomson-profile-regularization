"""Differentiable Thomson-scattering fit on time-resolved streaks.

Port of the original ThomsonScattering.fitting generalized to per-species
distribution models: each electron/ion species carries a
:class:`~.distributions.Distribution`, and its shape parameters (e.g. the
super-Gaussian exponent ``p`` → prefixes ``pe0`` / ``pi0``, a kappa index
``kappa`` → ``kappae0`` / ``kappai0``) join the universal moment parameters
(``n``, ``Te``/``ue``/``efract``, ``Ti``/``ui``/``ifract``) in the fit vector.

Public entry points:

- :class:`Param`      — tiny dataclass holding ``(value, min, max, vary)``.
- :func:`build_params` — assemble a ``{name: Param}`` dict for given
  per-species models, with the three-level specificity override pattern.
- :func:`run_fit_grad` — run a JAX + optax fit on a time-resolved Thomson
  streak. Returns ``(result, best_fit)``.
- :func:`compute_initial_fit` — evaluate the forward model at the initial
  guess (no fitting). Useful for diagnostic plots.

Models are chosen via ``measurement_settings["e_models"]`` /
``["i_models"]`` (lists of model specs — see
:func:`~.distributions.resolve_distribution`); both default to
``"super_gaussian"`` per species, which reproduces the original package's
parameters and results exactly.
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
from .distributions import resolve_models, shape_param_prefix
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

def _smax(a, b, w=0.01):
    """Smooth max: ``w·logaddexp(a/w, b/w)`` → max(a, b) as w → 0.

    Drop-in replacement for ``max(a, b)`` in constraint expressions when the
    result is sampled: the hard max has a gradient kink that stalls the
    HMC/MALA kernels (leapfrog energy errors at the kink force the step size
    down), while ``smax`` is C^∞ with a transition region of width ~w.
    Choose ``w`` small against the scale of ``a - b`` near the fit.
    """
    return w * jnp.logaddexp(a / w, b / w)


def _smin(a, b, w=0.01):
    """Smooth min: ``-smax(-a, -b, w)``."""
    return -_smax(-a, -b, w)


# Namespace exposed to constraint expressions. `min` / `max` are the binary
# jnp variants — matches the 2-arg form documented in the deck schema.
_CONSTRAINT_NS = {
    "min": jnp.minimum, "max": jnp.maximum,
    "smin": _smin, "smax": _smax,
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


# ─── prefix enumeration ──────────────────────────────────────────────────────

def _species_prefix_bases(e_models, i_models):
    """Per-species parameter bases as ``[(base, species_idx)]`` lists.

    Universal moment bases plus each model's shape-parameter bases. The base
    for shape param ``q`` is ``q + kind`` (e.g. ``"pe"``), so the full prefix
    ``f"{base}{s}"`` matches the original package's naming for super-Gaussians.
    """
    out = []
    for s, m in enumerate(e_models):
        for b in ("Te", "ue", "efract"):
            out.append((b, s))
        for q in m.shape_param_names:
            out.append((f"{q}e", s))
    for s, m in enumerate(i_models):
        for b in ("Ti", "ui", "ifract"):
            out.append((b, s))
        for q in m.shape_param_names:
            out.append((f"{q}i", s))
    return out


# ─── parameter building ──────────────────────────────────────────────────────

def build_params(e_models, i_models, Nt, params_settings=None, background_order=None):
    """Build a ``{name: Param}`` dict with the package's naming scheme.

    Parameters are keyed ``<var><species>_<time>`` (e.g. ``Te0_3``, ``pi1_0``,
    ``kappae0_2``); the per-shot density ``n_<time>`` has no species index.

    ``params_settings`` is a ``dict[str, dict]`` mapping parameter keys to
    Param-constructor kwargs (``value``, ``min``, ``max``, ``vary``). Keys use
    three-level specificity:

    - per-time, per-species: ``"Te0_3"``  → species 0 at t=3 only
    - per-species:           ``"Te0"``    → all times for species 0
    - global:                ``"Te"``     → all Te species at all times

    For ``n`` (no species index) the lookup checks ``"n_<t>"`` then ``"n"``.

    Shape parameters take their defaults (value/min/max) from the model's
    ``shape_param_defaults``; deck entries override as usual.

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

    _moment_defaults = {
        "Te": {"value": 100.0}, "ue": {"value": 0.0}, "efract": {"value": 1.0},
        "Ti": {"value": 100.0}, "ui": {"value": 0.0}, "ifract": {"value": 1.0},
    }

    for kind, models in (("e", e_models), ("i", i_models)):
        for s, model in enumerate(models):
            bases = ("Te", "ue", "efract") if kind == "e" else ("Ti", "ui", "ifract")
            for t in range(Nt):
                for b in bases:
                    p[f"{b}{s}_{t}"] = Param(**_lookup(b, s, t, _moment_defaults[b]))
                for q in model.shape_param_names:
                    default = dict(model.shape_param_defaults.get(q, {"value": 0.0}))
                    if "value" not in default:
                        default["value"] = 0.0
                    base = f"{q}{kind}"
                    p[f"{base}{s}_{t}"] = Param(**_lookup(base, s, t, default))

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


# ─── forward-model call ──────────────────────────────────────────────────────

def _make_forward_fn(measurement_settings, e_models, i_models):
    """Build the package-unit forward call closing over models + settings.

    Returns ``fwd(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg,
    instr_func_arr)`` where ``n`` is in cm^-3 and ``Te``/``Ti`` in eV
    (converted to m^-3 and K inside). ``instr_func_arr`` is explicit so the
    time-sharded path can substitute its per-shard slice; pass
    ``measurement_settings.get("instr_func_arr")`` for the plain path.

    The models are closed over (they are static Python objects); jit the
    returned function at the call site.
    """
    ms = measurement_settings

    def fwd(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg,
            instr_func_arr):
        return scattered_power_wavelength(
            n=n * 1e6,
            ue=ue, ui=ui,
            Te=Te * e / kB, Ti=Ti * e / kB,
            efract=efract, ifract=ifract,
            ion_z=ms["ion_z"],
            ion_a=ms["ion_a"],
            wavelengths=ms["wavelengths"],
            probe_wavelength=ms["probe_wavelength"],
            probe_vec=ms["probe_vec"],
            scatter_vec=ms["scatter_vec"],
            ue_dir=ms["ue_dir"],
            ui_dir=ms["ui_dir"],
            e_models=e_models,
            i_models=i_models,
            e_shapes=e_shapes,
            i_shapes=i_shapes,
            instr_func_arr=instr_func_arr,
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

    return fwd


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


# ─── time-axis sharding (intra-fit CPU parallelism) ──────────────────────────

def _pad_time_axis(moments, pad):
    """Pad each moment leaf's time axis (the last axis) by ``pad``, edge mode.

    ``moments`` is a pytree (the forward-args tuple, including the nested
    ``e_shapes`` / ``i_shapes``); ``None`` entries (e.g. absent background)
    pass through. Edge padding repeats the last real time column so the
    forward model stays finite on the padded columns; those columns carry NaN
    data and are masked out of the loss, and ``jnp.pad``'s VJP discards their
    gradients, so the real columns are unaffected.
    """
    if pad == 0:
        return moments

    def _pad(a):
        if jnp.ndim(a) == 1:
            return jnp.pad(a, (0, pad), mode="edge")
        return jnp.pad(a, ((0, 0), (0, pad)), mode="edge")

    return _jax.tree.map(_pad, moments)


def _make_sharded_nll(measurement_settings, Pkl_data, Pkl_var, Nt, n_dev,
                      has_bg, e_models, i_models):
    """Time-sharded data-fidelity term, fp-identical to
    ``_log_likelihood(fwd(...), Pkl_data, Pkl_var)``.

    The forward model is independent per time slice (the only cross-time
    coupling, the Tikhonov penalty, lives outside this term), so the Nt axis is
    sharded across ``n_dev`` CPU devices via ``shard_map``. Each device runs the
    full forward (all angles/wavelengths) for its block of time slices; only the
    masked chi^2 sum and the valid-pixel count are reduced across shards
    (``psum``). The Nt axis is padded up to a multiple of ``n_dev`` with NaN data
    columns, which mask out and contribute nothing — so the result matches the
    unsharded loss up to float64 summation order.

    The per-time instrument-response array ``instr_func_arr`` (Nk, Nt) must be
    sharded along time too — otherwise the forward's per-time IRF conv vmap sees
    a full-Nt IRF against an Nt-block spectrum and errors. We pad+shard it and
    pass the per-shard slice through the forward's explicit IRF argument.
    """
    from jax.sharding import Mesh, PartitionSpec as P
    try:
        # The (mesh=, in_specs=, out_specs=) signature validated in the spike.
        from jax.experimental.shard_map import shard_map as _shard_map
    except ImportError:                                  # removed in a future jax
        _shard_map = _jax.shard_map

    mesh = Mesh(np.asarray(_jax.devices()[:n_dev]), ("t",))
    Nt_pad = -(-int(Nt) // n_dev) * n_dev                 # ceil to multiple of n_dev
    pad = Nt_pad - int(Nt)
    data_p = jnp.pad(jnp.asarray(Pkl_data), ((0, 0), (0, pad)), constant_values=jnp.nan)
    var_p  = jnp.pad(jnp.asarray(Pkl_var),  ((0, 0), (0, pad)), constant_values=jnp.nan)

    # Per-time IRF (Nk, Nt): pad+shard along time. A 1-D (L,) uniform IRF needs
    # no sharding — it rides along in measurement_settings and applies whole to
    # each shard's time block. If absent or 1-D, pass a tiny dummy that the
    # forward ignores.
    _irf = measurement_settings.get("instr_func_arr", None)
    has_time_irf = _irf is not None and jnp.ndim(jnp.asarray(_irf)) == 2
    if has_time_irf:
        irf_p = jnp.pad(jnp.asarray(_irf), ((0, 0), (0, pad)), mode="edge")
    else:
        irf_p = jnp.zeros((1, Nt_pad), dtype=jnp.float64)

    fwd = _make_forward_fn(measurement_settings, e_models, i_models)

    PT, PST = P("t"), P(None, "t")          # (Nt,) and (·, Nt) arrays
    e_shape_specs = tuple(tuple(PT for _ in m.shape_param_names) for m in e_models)
    i_shape_specs = tuple(tuple(PT for _ in m.shape_param_names) for m in i_models)
    # n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract
    moment_specs = (PT, PST, PST, e_shape_specs, PST,
                    PST, PST, i_shape_specs, PST)

    def _core(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg,
              irf_local, data, var):
        fit = fwd(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg,
                  irf_local if has_time_irf
                  else measurement_settings.get("instr_func_arr", None))
        mask   = jnp.isnan(data) | jnp.isnan(fit)
        fit_s  = jnp.where(mask, 0.0, fit)
        data_s = jnp.where(mask, 0.0, data)
        var_s  = jnp.where(mask, 1.0, var)
        r = (fit_s - data_s) ** 2 / var_s
        sse = jnp.sum(r)
        cnt = jnp.sum((~mask).astype(r.dtype))
        return _jax.lax.psum(sse, "t"), _jax.lax.psum(cnt, "t")

    if has_bg:
        mapped = _shard_map(_core, mesh=mesh,
                            in_specs=moment_specs + (PST, PST, PST, PST),  # bg, irf, data, var
                            out_specs=(P(), P()))

        def sharded_nll(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg):
            m = _pad_time_axis(
                (n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg), pad)
            sse, cnt = mapped(*m, irf_p, data_p, var_p)
            return sse / cnt
    else:
        def _core_nobg(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract,
                       irf_local, data, var):
            return _core(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract,
                         None, irf_local, data, var)

        mapped = _shard_map(_core_nobg, mesh=mesh,
                            in_specs=moment_specs + (PST, PST, PST),  # irf, data, var
                            out_specs=(P(), P()))

        def sharded_nll(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg):
            m = _pad_time_axis(
                (n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract), pad)
            sse, cnt = mapped(*m, irf_p, data_p, var_p)
            return sse / cnt

    return sharded_nll


# ─── grad-problem assembly ──────────────────────────────────────────────────

def _build_penalty_list(penalty_settings, e_models, i_models, background_order):
    """Return ``list[(prefix, pset)]`` matched against parameter prefixes.

    Most-specific match wins (``Te0`` before ``Te``); ``n`` has no species idx.
    The for-loop is unrolled by jit at trace time, so penalty selection is free.
    """
    if penalty_settings is None:
        return []
    pairs = [("n", None)]
    if background_order is not None:
        pairs += [("bg", i) for i in range(background_order + 1)]
    pairs += _species_prefix_bases(e_models, i_models)
    out = []
    for base, s in pairs:
        prefix = base if s is None else f"{base}{s}"
        pset = penalty_settings.get(prefix)
        if pset is None:
            pset = penalty_settings.get(base)
        if pset is not None:
            out.append((prefix, pset))
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


def _log_det_jac_u(u, lower, upper, *, floor=1e-30):
    """Sum of per-coordinate log|dT_i/du_i| for the bijector built in
    :func:`_make_unconstrained_transforms`.

    Required for sampling targets where we want samples of ``x = T(u)`` to be
    distributed as ``p(x) ∝ exp(-V(x))``: change of variables means we draw
    ``u`` from ``exp(-V(T(u)) + log|det dT/du|)``. For optimization the term
    is irrelevant (MAP is invariant under reparameterization), so it lives
    here separately rather than inside ``objective_u``.

    Per-coordinate derivatives matching the branches in ``to_external_jax``:

    - bounded (arcsin):  x = lo + (hi-lo)/2 * (sin u + 1)
                         |dx/du| = (hi-lo)/2 * |cos u|
    - lo-only (sqrt):    x = lo - 1 + sqrt(u^2 + 1)
                         |dx/du| = |u| / sqrt(u^2 + 1)
    - hi-only (sqrt):    x = hi + 1 - sqrt(u^2 + 1)
                         |dx/du| = |u| / sqrt(u^2 + 1)
    - unbounded:         dx/du = 1, log term = 0

    ``floor`` keeps gradients finite if a sample lands exactly at u = 0 (sqrt
    branch) or u = ±π/2 (arcsin branch). The bijector already repels samples
    away from these points; the floor is belt-and-suspenders.
    """
    lo_fin = jnp.isfinite(lower)
    hi_fin = jnp.isfinite(upper)
    bounded = lo_fin & hi_fin
    one_sided = (lo_fin ^ hi_fin)  # XOR: exactly one finite bound

    log_bounded = (jnp.log((upper - lower) / 2.0)
                   + jnp.log(jnp.maximum(jnp.abs(jnp.cos(u)), floor)))
    log_one_sided = (jnp.log(jnp.maximum(jnp.abs(u), floor))
                     - 0.5 * jnp.log(u * u + 1.0))
    log_unbounded = jnp.zeros_like(u)

    contrib = jnp.where(bounded, log_bounded,
                        jnp.where(one_sided, log_one_sided, log_unbounded))
    return jnp.sum(contrib)


def _build_grad_problem(Pkl_data, Pkl_var, measurement_settings,
                        penalty_settings=None, params_settings=None,
                        constraints=None, extra_params=None, shard_time=None):
    """Set up the differentiable Thomson-fit problem in unconstrained space.

    Returns a SimpleNamespace bundling every closure and metadatum that
    ``run_fit_grad`` (and the posterior sampler) need.

    ``shard_time`` controls intra-fit time-axis sharding of the data-fidelity
    forward+grad: ``None`` (default) shards when the host exposes >1 XLA device,
    ``False`` forces it off, ``True`` forces it on when possible. Samplers pass
    ``False`` because the per-chain ``vmap`` does not compose cleanly with the
    ``shard_map`` inside the objective.
    """
    e_models, i_models = resolve_models(measurement_settings)
    Nelectrons = len(e_models)
    Nions = len(i_models)
    Nt = jnp.shape(Pkl_data)[1]
    background_order = measurement_settings.get("background_order", None)

    Pkl_data = jnp.array(Pkl_data)
    Pkl_var = jnp.array(Pkl_var)

    params = build_params(e_models, i_models, Nt, params_settings,
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
        penalty_settings, e_models, i_models, background_order
    )

    # Precompute per-prefix gather tables so _unpack / _build_params_dict can
    # assemble (Nt,) arrays in a single XLA gather + where instead of a Python
    # loop of Nt scalar indices. With Nt~651 the original loop made the JAX
    # trace huge, which dominated jit compile time (especially for sgld_lbfgs
    # which jits two separate step functions).
    def _make_gather_table(prefix):
        idx = np.zeros(Nt, dtype=np.int32)
        fxd = np.zeros(Nt, dtype=np.float64)
        free = np.zeros(Nt, dtype=bool)
        for t in range(Nt):
            key = f"{prefix}_{t}"
            if key in key_to_idx:
                idx[t] = key_to_idx[key]
                free[t] = True
            else:
                fxd[t] = fixed_vals[key]
        return jnp.asarray(idx), jnp.asarray(fxd), jnp.asarray(free)

    _species_bases = _species_prefix_bases(e_models, i_models)
    _all_prefixes = list(_extra_prefixes) + ["n"]
    _all_prefixes.extend(f"{base}{s}" for base, s in _species_bases)
    if background_order is not None:
        for _i in range(background_order + 1):
            _all_prefixes.append(f"bg{_i}")
    gather_tables = {
        pfx: _make_gather_table(pfx)
        for pfx in _all_prefixes
        if pfx not in _constraints
    }

    def _gather_prefix(x, prefix):
        idx_arr, fixed_arr, is_free = gather_tables[prefix]
        return jnp.where(is_free, x[idx_arr], fixed_arr)

    def _build_params_dict(x):
        """{prefix: (Nt,) array} for every parameter (free, fixed, constrained, extra).

        Extras are assembled first so constraints can reference them by name;
        thereafter prefixes are filled in declaration order, with constrained
        prefixes evaluated against everything accumulated so far.
        """
        p = {}
        for ep in _extra_prefixes:
            p[ep] = _gather_prefix(x, ep)

        def _fill(prefix):
            if prefix in _constraints:
                p[prefix] = _constraints[prefix](p)
            else:
                p[prefix] = _gather_prefix(x, prefix)

        _fill("n")
        for base, s in _species_bases:
            _fill(f"{base}{s}")
        if background_order is not None:
            for i in range(background_order + 1):
                _fill(f"bg{i}")
        return p

    def _args_from_params_dict(p):
        """Assemble the forward-args tuple from a {prefix: (Nt,)} dict."""
        n = p["n"]
        Te     = jnp.stack([p[f"Te{s}"]     for s in range(Nelectrons)])
        ue     = jnp.stack([p[f"ue{s}"]     for s in range(Nelectrons)])
        efract = jnp.stack([p[f"efract{s}"] for s in range(Nelectrons)])
        Ti     = jnp.stack([p[f"Ti{s}"]     for s in range(Nions)])
        ui     = jnp.stack([p[f"ui{s}"]     for s in range(Nions)])
        ifract = jnp.stack([p[f"ifract{s}"] for s in range(Nions)])
        e_shapes = tuple(
            tuple(p[shape_param_prefix(q, "e", s)] for q in m.shape_param_names)
            for s, m in enumerate(e_models)
        )
        i_shapes = tuple(
            tuple(p[shape_param_prefix(q, "i", s)] for q in m.shape_param_names)
            for s, m in enumerate(i_models)
        )
        bg = (jnp.stack([p[f"bg{i}"] for i in range(background_order + 1)])
              if background_order is not None else None)
        return n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg

    def _unpack_full(x):
        p = _build_params_dict(x)
        return _args_from_params_dict(p), p

    def _unpack(x):
        return _unpack_full(x)[0]

    fwd = _make_forward_fn(measurement_settings, e_models, i_models)
    _default_irf = measurement_settings.get("instr_func_arr", None)

    def _forward(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg):
        return fwd(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg,
                   _default_irf)

    # Intra-fit parallelism: when the host is exposed as multiple XLA devices
    # (THOMSON_CPU_DEVICES / --n-devices), shard the data-fidelity forward+grad
    # over the time axis. Numerically identical to the unsharded loss up to
    # float64 summation order; the Tikhonov penalty below stays unsharded.
    n_devices = _jax.device_count()
    want_shard = (n_devices > 1) if shard_time is None else bool(shard_time)
    use_time_shard = want_shard and n_devices > 1 and int(Nt) >= n_devices
    sharded_nll = (
        _make_sharded_nll(measurement_settings, Pkl_data, Pkl_var, int(Nt),
                          n_devices, has_bg=(background_order is not None),
                          e_models=e_models, i_models=i_models)
        if use_time_shard else None
    )

    # objective_flat must NOT be jit-decorated: jit(value_and_grad(f)) works,
    # but value_and_grad(jit(f)) cannot differentiate through the jit boundary.
    def objective_flat(x):
        args, p = _unpack_full(x)
        if use_time_shard:
            loss = sharded_nll(*args)
        else:
            fit = _forward(*args)
            loss = _log_likelihood(fit, Pkl_data, Pkl_var)
        for prefix, pset in penalty_list:
            loss = loss + _tikhonov_penalty(p[prefix], **pset)
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
        e_models=e_models, i_models=i_models,
        varying_keys=varying_keys,
        x0_np=x0_np, u0=u0,
        lower_np=lower_np, upper_np=upper_np,
        lower=lower, upper=upper,
        Nt=Nt,
        Pkl_data=Pkl_data, Pkl_var=Pkl_var,
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
    LBFGS gets stuck (typically when shape exponents are free and the loss
    has near-flat regions).

    Bounds are enforced by the unconstrained reparameterization (arcsin /
    sqrt / identity), so no clipping is needed inside the loop.

    Parameters
    ----------
    Pkl_data, Pkl_var, measurement_settings, penalty_settings, params_settings :
        See the package's deck schema and ``build_params``. Per-species
        distribution models come from ``measurement_settings["e_models"]`` /
        ``["i_models"]`` (default ``"super_gaussian"`` everywhere).
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
    e_models, i_models = resolve_models(measurement_settings)
    Nelectrons = len(e_models)
    Nions = len(i_models)
    background_order = measurement_settings.get("background_order", None)

    params = build_params(e_models, i_models, Nt, params_settings,
                          background_order=background_order)
    _expand_extra_params(params, extra_params, Nt)

    n      = extract_params_as_array(params, "n", Nt)
    Te     = jnp.stack([extract_params_as_array(params, f"Te{i}", Nt)
                        for i in range(Nelectrons)])
    ue     = jnp.stack([extract_params_as_array(params, f"ue{i}", Nt)
                        for i in range(Nelectrons)])
    efract = jnp.stack([extract_params_as_array(params, f"efract{i}", Nt)
                        for i in range(Nelectrons)])
    Ti     = jnp.stack([extract_params_as_array(params, f"Ti{i}", Nt)
                        for i in range(Nions)])
    ui     = jnp.stack([extract_params_as_array(params, f"ui{i}", Nt)
                        for i in range(Nions)])
    ifract = jnp.stack([extract_params_as_array(params, f"ifract{i}", Nt)
                        for i in range(Nions)])
    e_shapes = tuple(
        tuple(extract_params_as_array(params, shape_param_prefix(q, "e", s), Nt)
              for q in m.shape_param_names)
        for s, m in enumerate(e_models)
    )
    i_shapes = tuple(
        tuple(extract_params_as_array(params, shape_param_prefix(q, "i", s), Nt)
              for q in m.shape_param_names)
        for s, m in enumerate(i_models)
    )
    bg = None
    if background_order is not None:
        bg = jnp.stack([extract_params_as_array(params, f"bg{i}", Nt)
                        for i in range(background_order + 1)])

    fwd = _make_forward_fn(measurement_settings, e_models, i_models)
    return fwd(n, Te, ue, e_shapes, efract, Ti, ui, i_shapes, ifract, bg,
               measurement_settings.get("instr_func_arr", None))
