import re
import jax.numpy as jnp
from jax import jit
from lmfit import Parameters, Minimizer
from .utility import extract_params_as_array
from .forward import _scattered_power_wavelength
from scipy.constants import c, k as kB, epsilon_0, e, m_p


# Namespace exposed to constraint expressions evaluated in the grad backend.
# `min` / `max` are the binary jnp variants — they match the 2-arg form used
# in lmfit `expr=` strings, so the same expression text works for both backends.
_CONSTRAINT_NS = {
    "min": jnp.minimum, "max": jnp.maximum,
    "abs": jnp.abs, "where": jnp.where, "clip": jnp.clip,
    "sqrt": jnp.sqrt, "exp": jnp.exp, "log": jnp.log,
    "__builtins__": {},
}


def _compile_grad_constraints(constraints):
    """Convert {prefix: str} into {prefix: callable(p) -> (Nt,) array}.

    Pure-callable entries are passed through unchanged so existing Python-API
    callers keep working.
    """
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

_jitted_scattered_power_wavelength = jit(_scattered_power_wavelength,
                                         static_argnames=('normalization_type', 'notch', 'irf_normalization', 'gain_mode'))

#Now for building the fitter

#Log likelihood of the fit being measured out of the data, obtained by averaging over the residuals
#The reason I average and not sum is to make the regularization weights not depend on number of timesteps
def _log_likelihood(fit, data, variance):
    return jnp.nanmean((fit - data) ** 2 / variance)


#no input sanitization here
#param_profile should have shape (Nt, 1) while everything else should be
def _tikhonov_penalty(param_array,
                     profile_axis,
                     lambda_weights,
                     thresholds,
                     relative = True,
                     norm_scale = 1,
                     monotonic = 0):
    if len(lambda_weights) != 3:
        raise ValueError(f"lambda_weights must have exactly 3 elements, got {len(lambda_weights)}")

    penalty = 0

    if not hasattr(norm_scale, '__len__'):
        norm_scale = [norm_scale] * 3
    if not hasattr(monotonic, '__len__'):
        monotonic = [monotonic] * 3

    dt = jnp.diff(profile_axis)            # (N-1,)
    dt_mid = (dt[:-1] + dt[1:]) / 2        # (N-2,)
    d1 = jnp.diff(param_array) / dt        # (N-1,)
    d2 = jnp.diff(d1) / dt_mid             # (N-2,)
    derivs = [param_array, d1, d2]

    # relative_factor trimmed to match each derivative's length via local averaging
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
        penalty += (lambda_weights[order]
                    * jnp.mean(adjusted_deriv**2))

    return penalty

# the prior distribution, defined by the Tikhonov penalties
def _log_prior(params, Nindices, penalty_settings):
    # penalty_settings keys can be species-specific ("Ti0", "Ti1") or
    # global ("Ti"), which applies to all species sharing that base name.
    # For each param prefix found in params, we look up penalty settings
    # from most-specific to least-specific, mirroring build_minimal_params.
    def _lookup(prefix):
        base = prefix.rstrip('0123456789')
        if prefix in penalty_settings:
            return penalty_settings[prefix]
        if base in penalty_settings:
            return penalty_settings[base]
        return None

    # Collect unique {var}{species} prefixes by stripping the trailing _{t}
    prefixes = dict.fromkeys(key.rsplit("_", 1)[0] for key in params)

    total_penalty = 0
    for prefix in prefixes:
        settings = _lookup(prefix)
        if settings is None:
            continue
        param_array = extract_params_as_array(params, prefix, Nindices)
        #print(prefix, settings)
        current_penalty = _tikhonov_penalty(param_array, **settings)
        total_penalty += current_penalty
        #print(prefix, current_penalty)
    return total_penalty

def _compute_fit(params, measurement_settings):
    """Evaluate the forward model at the given params and return scattered_power_wavelength."""
    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    Nt = len([k for k in params if k.startswith("n_")])

    n = extract_params_as_array(params, "n", Nt)

    Te = jnp.stack([extract_params_as_array(params, f"Te{i}", Nt) for i in range(Nelectrons)])
    ue = jnp.stack([extract_params_as_array(params, f"ue{i}", Nt) for i in range(Nelectrons)])
    pe = jnp.stack([extract_params_as_array(params, f"pe{i}", Nt) for i in range(Nelectrons)])
    efract = jnp.stack([extract_params_as_array(params, f"efract{i}", Nt) for i in range(Nelectrons)])

    Ti = jnp.stack([extract_params_as_array(params, f"Ti{i}", Nt) for i in range(Nions)])
    ui = jnp.stack([extract_params_as_array(params, f"ui{i}", Nt) for i in range(Nions)])
    pi = jnp.stack([extract_params_as_array(params, f"pi{i}", Nt) for i in range(Nions)])
    ifract = jnp.stack([extract_params_as_array(params, f"ifract{i}", Nt) for i in range(Nions)])

    background_order = measurement_settings.get("background_order", None)
    if background_order is not None:
        background_coefs = jnp.stack(
            [extract_params_as_array(params, f"bg{i}", Nt) for i in range(background_order + 1)]
        )
    else:
        background_coefs = None

    return _jitted_scattered_power_wavelength(
        n=n * 1e6,
        ue=ue,
        ui=ui,
        Te=Te * e / kB,
        Ti=Ti * e / kB,
        pe=pe,
        pi=pi,
        efract=efract,
        ifract=ifract,
        ion_z=measurement_settings["ion_z"],
        ion_a=measurement_settings["ion_a"],
        wavelengths=measurement_settings["wavelengths"],
        probe_wavelength=measurement_settings["probe_wavelength"],
        probe_vec=measurement_settings["probe_vec"],
        scatter_vec=measurement_settings["scatter_vec"],
        ue_dir=measurement_settings["ue_dir"],
        ui_dir=measurement_settings["ui_dir"],
        instr_func_arr=measurement_settings.get("instr_func_arr", None),
        irf_normalization=measurement_settings.get("irf_normalization", "area"),
        throughput=measurement_settings.get("throughput", None),
        aperture_weights=measurement_settings.get("aperture_weights", None),
        background_coefs=background_coefs,
        normalization_type=measurement_settings.get("normalization_type", "max"),
        normalization_scale=measurement_settings.get("normalization_scale", 1),
        notch=measurement_settings.get("notch", None),
        probe_intensity=measurement_settings.get("probe_intensity", 0.0),
        probe_diameter=measurement_settings.get("probe_diameter", 1.0),
        pol_p_fraction=measurement_settings.get("pol_p_fraction", 1.0),
        gain_mode=measurement_settings.get("gain_mode", "off"),
    )


#Now define the full objective function which sums the log_likelihood + log_prior to get the log posterior
def _log_posterior(params, Pkl_data, Pkl_var, measurement_settings, penalty_settings, use_penalty=True):
    Nt = jnp.shape(Pkl_data)[1]
    fit = _compute_fit(params, measurement_settings)
    ll = _log_likelihood(fit, Pkl_data, Pkl_var)
    prior = _log_prior(params, Nt, penalty_settings) if (use_penalty and penalty_settings is not None) else 0
    return ll + prior


def run_fit(
    Pkl_data,
    Pkl_var,
    measurement_settings,
    penalty_settings=None,
    params_settings=None,
    fit_settings=None,
    extra_params=None,
    constraints=None,
    progress=False,
):
    """Run the regularized Thomson scattering fit on a streak.

    Parameters
    ----------
    Pkl_data : array (Nk, Nt)
        Measured scattered power spectrum (wavelength × time).
    Pkl_var : array (Nk, Nt)
        Variance of the measured data.
    measurement_settings : dict
        Static geometry and composition settings. Required keys:
        Nelectrons, ion_z, ion_a, wavelengths, probe_wavelength,
        probe_vec, scatter_vec, ue_dir, ui_dir.
        Optional: instr_func_arr, normalization_type, normalization_scale.
    penalty_settings : dict or None
        Tikhonov regularization settings keyed by parameter name.
        Passed directly to _log_prior. None disables regularization.
    params_settings : dict or None
        Per-parameter lmfit kwargs passed into build_params. Supports the same
        three-level key specificity ("Te", "Te0", "Te0_3"). Values are dicts of
        lmfit.Parameters.add() kwargs, e.g. {"value": 100.0, "vary": False, "min": 0}.
        If None, build_params defaults are used.
    fit_settings : dict or None
        Optimizer settings. Supported keys:
          - 'method' (str, default 'nelder'): method string for lmfit Minimizer
          - any other keys are passed through as kwargs to Minimizer.minimize()
    extra_params : list of dict or None
        Extra parameters to inject into the fitting (e.g. dummy variables for expr constraints).
        Each dict must have a "name" key, plus any valid lmfit.Parameters.add() kwargs.
        The parameter will be replicated across all Nt time slices as {name}_0, {name}_1, etc.
        If an "expr" string is provided, {t} substitution is applied: if {t} is not present,
        _{t} is appended before calling .format(t=t).
    constraints : dict or None
        Mapping of parameter prefix (e.g. ``"ifract1"``) to a string expression
        written in terms of other prefix names (no ``_{t}`` suffix). Each entry
        is replicated across all Nt time slices: prefix names in the expression
        are rewritten to their per-time form before being assigned to the
        target parameter's ``expr`` attribute. Same syntax as ``run_fit_grad``,
        so a deck's ``[constraints]`` section drives both backends.
    progress : bool
        If True, display a tqdm progress bar updated each iteration showing
        the current objective value.

    Returns
    -------
    result : lmfit.MinimizerResult
        Full result from lmfit, including result.params with the best-fit
        values and result.success indicating convergence.
    best_fit : jnp.array, shape (Nk, Nt)
        Scattered power spectrum evaluated at the best-fit parameters.
    """
    if fit_settings is None:
        fit_settings = {}
    fit_settings = dict(fit_settings)  # copy to avoid mutating caller's dict
    method = fit_settings.pop("method", "nelder")

    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    Nt = jnp.shape(Pkl_data)[1]

    # Create params and add extra parameters first (before main params)
    params = Parameters()
    if extra_params is not None:
        for extra_def in extra_params:
            # Extract the "name" key and other lmfit.Parameters.add() kwargs
            extra_def_copy = dict(extra_def)
            param_name = extra_def_copy.pop("name")

            # Replicate across all time slices
            for t in range(Nt):
                # Apply {t} substitution to expr if present
                kwargs = dict(extra_def_copy)
                if "expr" in kwargs:
                    expr = kwargs["expr"]
                    if "{t}" not in expr:
                        expr = expr + "_{t}"
                    kwargs["expr"] = expr.format(t=t)

                # Add the parameter
                params.add(f"{param_name}_{t}", **kwargs)

    # Build and add main parameters
    main_params = build_params(
        Nelectrons, Nions, Nt, params_settings,
        background_order=measurement_settings.get("background_order", None),
    )
    for key, val in main_params.items():
        params[key] = val

    # Apply [constraints] by rewriting bare prefix names to their per-time form
    # and assigning to the target parameter's expr. Done after extras + main
    # params are registered so expressions may reference either.
    if constraints:
        known_prefixes = {k.rsplit("_", 1)[0] for k in params.keys()}
        # Sort longest-first so e.g. "ifract10" matches before "ifract1".
        prefix_pat = re.compile(
            r"\b(" + "|".join(
                re.escape(p) for p in sorted(known_prefixes, key=len, reverse=True)
            ) + r")\b"
        )
        for prefix, expr in constraints.items():
            for t in range(Nt):
                target = f"{prefix}_{t}"
                if target not in params:
                    raise KeyError(
                        f"[constraints] target {target!r} not found in params. "
                        f"Constraint prefix {prefix!r} must match a known parameter."
                    )
                params[target].expr = prefix_pat.sub(
                    lambda m, _t=t: f"{m.group(1)}_{_t}", expr
                )

    Pkl_data = jnp.array(Pkl_data)
    Pkl_var = jnp.array(Pkl_var)

    def objective(p):
        return _log_posterior(p, Pkl_data, Pkl_var, measurement_settings, penalty_settings)

    iter_cb = None
    if progress:
        from tqdm.auto import tqdm
        from collections import deque
        bar = tqdm(desc=f"run_fit ({method})", unit="iter")
        window = deque(maxlen=100)

        def iter_cb(_p, _itr, resid):
            window.append(float(resid))
            bar.update(1)
            postfix = {"obj": f"{float(resid):.4g}"}
            if len(window) == window.maxlen:
                abs_improvement = window[0] - min(window)
                rel_improvement = abs_improvement / (abs(window[0]) + 1e-300)
                postfix["d_obj"] = f"{abs_improvement:.2e}"
                postfix["d_obj_rel"] = f"{rel_improvement:.2e}"
            bar.set_postfix(postfix)

    minner = Minimizer(objective, params, nan_policy="omit", iter_cb=iter_cb)
    try:
        result = minner.minimize(method=method, **fit_settings)
    finally:
        if progress:
            bar.close()

    best_fit = _compute_fit(result.params, measurement_settings)

    return result, best_fit


def _scale_penalty_settings(penalty_settings, weight_scale, cutoff_scale):
    """Return a copy of penalty_settings with lambda_weights and thresholds scaled."""
    scaled = {}
    for key, settings in penalty_settings.items():
        s = dict(settings)
        s["lambda_weights"] = [w * weight_scale for w in settings["lambda_weights"]]
        s["thresholds"] = [t * cutoff_scale for t in settings["thresholds"]]
        scaled[key] = s
    return scaled


def chi2_vary_tikhonov(
    Pkl_data,
    Pkl_var,
    measurement_settings,
    penalty_settings,
    weight_scales,
    cutoff_scales,
    params_settings=None,
    fit_settings=None,
    extra_params=None,
    progress=False,
):
    """Scan Tikhonov weights and thresholds and return chi2 on a 2D grid.

    For each (weight_scale, cutoff_scale) pair, all lambda_weights in
    penalty_settings are multiplied by weight_scale and all thresholds by
    cutoff_scale. A full fit is run at each grid point and the log likelihood
    (chi2, data fidelity only — no regularization penalty) is recorded.

    The tightest penalties that don't significantly inflate chi2 relative to
    the unregularized fit are the most physically motivated.

    Parameters
    ----------
    Pkl_data, Pkl_var, measurement_settings, params_settings, fit_settings, extra_params :
        Same as run_fit.
    penalty_settings : dict
        Base penalty settings (must not be None).
    weight_scales : array-like
        Multiplicative scale factors applied to all lambda_weights.
    cutoff_scales : array-like
        Multiplicative scale factors applied to all thresholds.

    Returns
    -------
    chi2 : jnp.array, shape (len(weight_scales), len(cutoff_scales))
        Log likelihood (chi2) at the best-fit parameters for each grid point.
    params_grid : list of list of dict
        Fitted parameters from each fit, organized as params_grid[i][j]
        contains the parameters dict for weight_scales[i], cutoff_scales[j].
    """
    Pkl_data = jnp.array(Pkl_data)
    Pkl_var = jnp.array(Pkl_var)

    chi2_grid = jnp.zeros((len(weight_scales), len(cutoff_scales)))
    params_grid = [[None for _ in cutoff_scales] for _ in weight_scales]

    for i, ws in enumerate(weight_scales):
        for j, cs in enumerate(cutoff_scales):
            scaled = _scale_penalty_settings(penalty_settings, ws, cs)
            result, best_fit = run_fit(
                Pkl_data, Pkl_var, measurement_settings,
                penalty_settings=scaled,
                params_settings=params_settings,
                fit_settings=fit_settings,
                extra_params=extra_params,
                progress=progress,
            )

            chi2_val = _log_likelihood(best_fit, Pkl_data, Pkl_var)

            chi2_grid = chi2_grid.at[i, j].set(chi2_val)
            params_grid[i][j] = dict(result.params)
    return chi2_grid, params_grid


def build_params(Nelectrons, Nions, Nt, params_settings=None, background_order=None):
    """Build an lmfit.Parameters object with the naming scheme used by the fitter.

    Parameters created follow the pattern <var><species>_<time>, e.g. `Te0_0`, `Ti1_3`.

    params_settings is a dict mapping parameter keys to dicts of lmfit.Parameters.add()
    kwargs (e.g. {"value": 100.0, "vary": False, "min": 0}). Keys use the same
    three-level specificity as penalty_settings:
      - per-time, per-species: "Te0_3" -> species 0 at t=3 only
      - per-species:           "Te0"   -> all times for species 0
      - global:                "Te"    -> all Te species at all times
    For `n` (no species index), the lookup checks "n_<t>" (time-specific) then "n" (global).
    User-supplied kwargs are merged on top of per-variable defaults, so partial
    dicts like {"vary": False} are fine — the default value is still applied.

    If background_order is not None, polynomial background coefs `bg{i}_{t}` for
    i in range(background_order+1) are added; these enter the forward model as a
    polynomial in (lam - lam0)/lam0 and the existing Tikhonov machinery regularizes
    their time evolution per-prefix ("bg0", "bg1", ...).

    Returns an lmfit.Parameters instance.
    """
    p = Parameters()
    if params_settings is None:
        params_settings = {}

    def _lookup(base, species, t, default):
        # check most-specific to least-specific, merge user settings onto defaults
        # if species is None, skip the species-level keys (used for `n` which has no species)
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
        merged = {**default, **user}
        if "expr" in merged:
            expr = merged["expr"]
            if "{t}" not in expr:
                expr = expr + "_{t}"
            merged["expr"] = expr.format(t=t)
        return merged

    # total electron density time-series `n_{t}` (no species index)
    for t in range(Nt):
        p.add(f"n_{t}", **_lookup("n", None, t, {"value": 1e20}))

    # electron moments
    for s in range(Nelectrons):
        for t in range(Nt):
            p.add(f"Te{s}_{t}", **_lookup("Te", s, t, {"value": 100.0}))
            p.add(f"ue{s}_{t}", **_lookup("ue", s, t, {"value": 0.0}))
            p.add(f"pe{s}_{t}", **_lookup("pe", s, t, {"value": 2.0}))
            p.add(f"efract{s}_{t}", **_lookup("efract", s, t, {"value": 1.0}))

    # ion moments
    for s in range(Nions):
        for t in range(Nt):
            p.add(f"Ti{s}_{t}", **_lookup("Ti", s, t, {"value": 100.0}))
            p.add(f"ui{s}_{t}", **_lookup("ui", s, t, {"value": 0.0}))
            p.add(f"pi{s}_{t}", **_lookup("pi", s, t, {"value": 2.0}))
            p.add(f"ifract{s}_{t}", **_lookup("ifract", s, t, {"value": 1.0}))

    # polynomial background coefs `bg{i}_{t}` (signed; default 0)
    if background_order is not None:
        for i in range(background_order + 1):
            for t in range(Nt):
                p.add(f"bg{i}_{t}", **_lookup("bg", i, t, {"value": 0.0}))

    return p


def run_fit_grad(
    Pkl_data,
    Pkl_var,
    measurement_settings,
    penalty_settings=None,
    params_settings=None,
    constraints=None,
    extra_params=None,
    fit_settings=None,
    progress=False,
):
    """Run the Thomson scattering fit using JAX autodiff + optax.

    Computes exact gradients via jax.value_and_grad in a single backward pass,
    then steps with an optax optimizer. Defaults to optax.lbfgs (with built-in
    Zoom line search) — the closest analogue to scipy's L-BFGS-B and the right
    default for Thomson fits where the loss is mostly quadratic near the
    minimum. Adam/AdamW are also available for cases where lbfgs gets stuck
    (typically when pe/pi are free and gammaincc produces near-flat regions).

    Bounds are enforced by clipping after each update.

    Parameters
    ----------
    Pkl_data, Pkl_var, measurement_settings, penalty_settings, params_settings :
        Same semantics as run_fit.
    constraints : dict or None
        Equality-style reparameterization. Keys are parameter prefixes (e.g.
        "ifract1"); values are EITHER:
          - a string expression evaluated against (Nt,) jnp arrays, with names
            ``min``, ``max``, ``abs``, ``where``, ``clip``, ``sqrt``, ``exp``,
            ``log`` available — same expression text works for the lmfit backend.
          - a callable receiving the accumulated dict p of {prefix: (Nt,) array}
            and returning a (Nt,) array.
        Constrained prefixes are excluded from the free-variable vector x and
        are evaluated after all other prefixes (and any extra_params) are
        assembled, so expressions may reference any of them by name.

        Example::

            constraints = {
                "ifract1": "1 - ifract0",
                "ue1":     lambda p: -p["ue0"],
                "ifract2": "min(1 - ifract0 - ifract1, ifract2_dummy)",
            }
    extra_params : list of dict or None
        Free dummy parameters injected into the fit. Each dict needs a "name"
        key plus any lmfit add() kwargs (value, min, max, vary). Replicated as
        ``{name}_0, {name}_1, ...`` across Nt time steps and made available to
        constraint expressions under the bare name (e.g. ``ifract1_dummy``).
    fit_settings : dict or None
        Optimizer settings. Recognized keys:
          - 'optimizer' (str, default 'lbfgs'): 'lbfgs' | 'adam' | 'adamw'
          - 'lr' / 'learning_rate' (float, default 1e-2): step size for adam/adamw
          - 'max_iter' (int, default 1000): hard iteration cap
          - 'tol' (float, default 1e-8): converged when relative loss
            improvement over a 100-iteration window falls below tol
        All other keys flow into the optax optimizer constructor (e.g.
        memory_size for lbfgs, weight_decay for adamw).
    progress : bool
        If True, display a tqdm progress bar.

    Returns
    -------
    result : types.SimpleNamespace
        result.x            : np.ndarray of best-fit parameter values
        result.varying_keys : list[str], parameter names matching result.x
        result.fun          : float, final loss
        result.nit          : int, number of iterations completed
        result.success      : bool, True if window-based tolerance was reached
                              before max_iter

    best_fit : jnp.array, shape (Nk, Nt)

    Notes
    -----
    gammaincc differentiation: if pe/pi are free, JAX must have a registered
    VJP for jax.scipy.special.gammaincc. If autodiff raises NotImplementedError,
    fix pe and pi as non-varying in params_settings.

    First call triggers JIT compilation; subsequent calls with the same static
    structure are fast.
    """
    import numpy as np
    import optax
    from types import SimpleNamespace
    from collections import deque
    from jax import value_and_grad

    if fit_settings is None:
        fit_settings = {}
    fit_settings = dict(fit_settings)
    fit_settings.pop("method", None)
    optimizer_name = fit_settings.pop("optimizer", "lbfgs")
    lr = fit_settings.pop("lr", fit_settings.pop("learning_rate", 1e-2))
    max_iter = fit_settings.pop("max_iter", 1000)
    tol = fit_settings.pop("tol", 1e-8)

    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    Nt = jnp.shape(Pkl_data)[1]
    background_order = measurement_settings.get("background_order", None)

    Pkl_data = jnp.array(Pkl_data)
    Pkl_var = jnp.array(Pkl_var)

    params = build_params(
        Nelectrons, Nions, Nt, params_settings,
        background_order=background_order,
    )

    # Inject extra dummy parameters (replicated across time slices) so they
    # behave like ordinary free variables and can be referenced by name in
    # constraint expressions (e.g. ``ifract1_dummy``).
    _extra_prefixes = []
    if extra_params is not None:
        for extra_def in extra_params:
            extra_def_copy = dict(extra_def)
            param_name = extra_def_copy.pop("name")
            _extra_prefixes.append(param_name)
            for t in range(Nt):
                kwargs = dict(extra_def_copy)
                if "expr" in kwargs:
                    expr = kwargs["expr"]
                    if "{t}" not in expr:
                        expr = expr + "_{t}"
                    kwargs["expr"] = expr.format(t=t)
                params.add(f"{param_name}_{t}", **kwargs)

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

    x0 = np.array([params[k].value for k in varying_keys], dtype=np.float64)
    lower = jnp.array([params[k].min for k in varying_keys])
    upper = jnp.array([params[k].max for k in varying_keys])

    # Build penalty list at construction time; the for-loop is unrolled by jit.
    penalty_list = []
    if penalty_settings is not None:
        bg_block = (
            [("bg", background_order + 1)] if background_order is not None else []
        )
        for base, n_sp in (
            [("n", 1)]
            + bg_block
            + [(b, Nelectrons) for b in ("Te", "ue", "pe", "efract")]
            + [(b, Nions)      for b in ("Ti", "ui", "pi", "ifract")]
        ):
            for s in range(n_sp):
                prefix = base if base == "n" else f"{base}{s}"
                pset = penalty_settings.get(prefix)
                if pset is None:
                    pset = penalty_settings.get(base)
                if pset is not None:
                    penalty_list.append((base, None if base == "n" else s, pset))

    def _get(x, key):
        if key in key_to_idx:
            return x[key_to_idx[key]]
        return jnp.array(fixed_vals[key])

    def _unpack(x):
        # p accumulates {prefix: (Nt,) array} so constraint lambdas can
        # reference previously assembled parameters by name.
        p = {}

        # Assemble extra (dummy) parameters first so constraints can reference
        # them by their bare name regardless of where the constraint sits in
        # the standard prefix order.
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

        n      = _row("n",  None)
        Te     = jnp.stack([_row("Te",     s) for s in range(Nelectrons)])
        ue     = jnp.stack([_row("ue",     s) for s in range(Nelectrons)])
        pe     = jnp.stack([_row("pe",     s) for s in range(Nelectrons)])
        efract = jnp.stack([_row("efract", s) for s in range(Nelectrons)])
        Ti     = jnp.stack([_row("Ti",     s) for s in range(Nions)])
        ui     = jnp.stack([_row("ui",     s) for s in range(Nions)])
        pi_arr = jnp.stack([_row("pi",     s) for s in range(Nions)])
        ifract = jnp.stack([_row("ifract", s) for s in range(Nions)])
        if background_order is not None:
            bg = jnp.stack([_row("bg", i) for i in range(background_order + 1)])
        else:
            bg = None
        return n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg

    def _forward(n, Te, ue, pe, efract, Ti, ui, pi_arr, ifract, bg):
        return _jitted_scattered_power_wavelength(
            n=n * 1e6,
            ue=ue, ui=ui,
            Te=Te * e / kB, Ti=Ti * e / kB,
            pe=pe, pi=pi_arr,
            efract=efract, ifract=ifract,
            ion_z=measurement_settings["ion_z"],
            ion_a=measurement_settings["ion_a"],
            wavelengths=measurement_settings["wavelengths"],
            probe_wavelength=measurement_settings["probe_wavelength"],
            probe_vec=measurement_settings["probe_vec"],
            scatter_vec=measurement_settings["scatter_vec"],
            ue_dir=measurement_settings["ue_dir"],
            ui_dir=measurement_settings["ui_dir"],
            instr_func_arr=measurement_settings.get("instr_func_arr", None),
            irf_normalization=measurement_settings.get("irf_normalization", "area"),
            throughput=measurement_settings.get("throughput", None),
            aperture_weights=measurement_settings.get("aperture_weights", None),
            background_coefs=bg,
            normalization_type=measurement_settings.get("normalization_type", "max"),
            normalization_scale=measurement_settings.get("normalization_scale", 1),
            notch=measurement_settings.get("notch", None),
            probe_intensity=measurement_settings.get("probe_intensity", 0.0),
            probe_diameter=measurement_settings.get("probe_diameter", 1.0),
            pol_p_fraction=measurement_settings.get("pol_p_fraction", 1.0),
            gain_mode=measurement_settings.get("gain_mode", "off"),
        )

    # objective_flat must NOT be jit-decorated: jit(value_and_grad(f)) works,
    # but value_and_grad(jit(f)) cannot differentiate through the jit boundary.
    # optax.lbfgs's line search calls value_fn internally; that path is
    # jit-compiled by optax once the outer step is jit'd.
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

    val_and_grad_fn = jit(value_and_grad(objective_flat))

    # Build the optax optimizer; remaining fit_settings flow through as kwargs.
    if optimizer_name == "lbfgs":
        opt = optax.lbfgs(**fit_settings)
        needs_value_fn = True
    elif optimizer_name == "adam":
        opt = optax.adam(learning_rate=lr, **fit_settings)
        needs_value_fn = False
    elif optimizer_name == "adamw":
        opt = optax.adamw(learning_rate=lr, **fit_settings)
        needs_value_fn = False
    else:
        raise ValueError(
            f"Unknown optimizer {optimizer_name!r}. Use 'lbfgs', 'adam', or 'adamw'."
        )

    x = jnp.array(x0)
    opt_state = opt.init(x)

    if needs_value_fn:
        @jit
        def step(x, opt_state):
            val, grad = val_and_grad_fn(x)
            updates, new_opt_state = opt.update(
                grad, opt_state, x,
                value=val, grad=grad, value_fn=objective_flat,
            )
            new_x = jnp.clip(optax.apply_updates(x, updates), lower, upper)
            return new_x, new_opt_state, val
    else:
        @jit
        def step(x, opt_state):
            val, grad = val_and_grad_fn(x)
            updates, new_opt_state = opt.update(grad, opt_state, x)
            new_x = jnp.clip(optax.apply_updates(x, updates), lower, upper)
            return new_x, new_opt_state, val

    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(desc=f"run_fit_grad ({optimizer_name})", unit="iter", total=max_iter)
    else:
        bar = None
    window = deque(maxlen=100)

    converged = False
    val_f = float("inf")
    nit = 0
    try:
        for i in range(max_iter):
            x, opt_state, val = step(x, opt_state)
            nit = i + 1
            val_f = float(val)
            window.append(val_f)
            if bar is not None:
                bar.update(1)
                postfix = {"obj": f"{val_f:.4g}"}
                if len(window) == window.maxlen:
                    rel = (window[0] - min(window)) / (abs(window[0]) + 1e-300)
                    postfix["d_obj_rel"] = f"{rel:.2e}"
                bar.set_postfix(postfix)
            # window-based convergence: relative improvement over the last
            # 100 iterations below tol => stop.
            if len(window) == window.maxlen:
                rel = (window[0] - min(window)) / (abs(window[0]) + 1e-300)
                if rel < tol:
                    converged = True
                    break
    finally:
        if bar is not None:
            bar.close()

    # Final loss at the final x (val above is the loss *before* the last step).
    final_val, _ = val_and_grad_fn(x)

    result = SimpleNamespace(
        x=np.array(x),
        fun=float(final_val),
        nit=nit,
        success=converged,
        varying_keys=varying_keys,
    )
    best_fit = _forward(*_unpack(jnp.array(result.x)))

    return result, best_fit
