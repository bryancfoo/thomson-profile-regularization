import jax.numpy as jnp
from lmfit import Parameters, Minimizer
from utility import extract_params_as_array
from forward import _scattered_power_wavelength
from scipy.constants import c, k as kB, epsilon_0, e, m_p

#Now for building the fitter

#Log likelihood of the fit being measured out of the data, obtained by averaging over the residuals
#The reason I average and not sum is to make the regularization weights not depend on number of timesteps
def _log_likelihood(fit, data, variance):
    return jnp.mean((fit - data) ** 2 / variance)


#no input sanitization here
#param_profile should have shape (Nt, 1) while everything else should be
def _tikhonov_penalty(param_array,
                     profile_axis,
                     lambda_weights,
                     thresholds,
                     relative = True,
                     norm_scale = 1):
    penalty = 0
    deriv = param_array.copy() #the current derivative being penalized. Starts at 0th

    #If relative, the norm_scale is also relative
    norm_scale = norm_scale * (1 + relative * (jnp.abs(param_array) - 1))

    for order in range(len(lambda_weights)):
        # penalty only kicks in at the threshold
        # also scale by relative and also norm_scale as needed
        current_threshold = thresholds[order] / (1 + relative * (jnp.abs(param_array) - 1))
        adjusted_deriv = jnp.maximum(0, jnp.abs(deriv) - current_threshold) / norm_scale
        penalty += (lambda_weights[order] #weight of the penalty
                    * jnp.mean(adjusted_deriv**2))

        #Now take the next derivative
        deriv = jnp.gradient(deriv, profile_axis)

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
        total_penalty += _tikhonov_penalty(param_array, **settings)
    return total_penalty

#Now define the full objective function which sums the log_likelihood + log_prior to get the log posterior
def _log_posterior(params, Pkl_data, Pkl_var, measurement_settings, penalty_settings, use_penalty = True):
    #First count the number of species for both electrons and ions
    Nelectrons = measurement_settings["Nelectrons"] #for the electrons I'll take this as an explicit input
    Nions = len(measurement_settings["ion_z"])
    Nt = jnp.shape(Pkl_data)[1] #number of time steps is the second dimension of the data
    # Build parameter arrays for electrons and ions using extract_params_as_array
    # Total electron density `n` is time-series (no species index): n_{t}

    # Extract total electron density time-series from params
    n = extract_params_as_array(params, "n", Nt)

    Te = jnp.zeros((Nelectrons, Nt))
    ue = jnp.zeros((Nelectrons, Nt))
    pe = jnp.zeros((Nelectrons, Nt))
    efract = jnp.zeros((Nelectrons, Nt))
    for i in range(Nelectrons):
        Te = Te.at[i, :].set(extract_params_as_array(params, f"Te{i}", Nt))
        ue = ue.at[i, :].set(extract_params_as_array(params, f"ue{i}", Nt))
        pe = pe.at[i, :].set(extract_params_as_array(params, f"pe{i}", Nt))
        efract = efract.at[i, :].set(extract_params_as_array(params, f"efract{i}", Nt))

    Ti = jnp.zeros((Nions, Nt))
    ui = jnp.zeros((Nions, Nt))
    pi = jnp.zeros((Nions, Nt))
    ifract = jnp.zeros((Nions, Nt))
    for i in range(Nions):
        Ti = Ti.at[i, :].set(extract_params_as_array(params, f"Ti{i}", Nt))
        ui = ui.at[i, :].set(extract_params_as_array(params, f"ui{i}", Nt))
        pi = pi.at[i, :].set(extract_params_as_array(params, f"pi{i}", Nt))
        ifract = ifract.at[i, :].set(extract_params_as_array(params, f"ifract{i}", Nt))

    # Read measurement/static settings required by the forward model
    # efract and ifract are now fitted moments (in `params`), so read
    # ion composition and optics from measurement_settings
    ion_z = measurement_settings["ion_z"]
    ion_a = measurement_settings["ion_a"]
    wavelengths = measurement_settings["wavelengths"]
    probe_wavelength = measurement_settings["probe_wavelength"]
    probe_vec = measurement_settings["probe_vec"]
    scatter_vec = measurement_settings["scatter_vec"]
    ue_dir = measurement_settings["ue_dir"]
    ui_dir = measurement_settings["ui_dir"]
    instr_func_arr = measurement_settings.get("instr_func_arr", None)
    normalization_type = measurement_settings.get("normalization_type", "integral")
    normalization_scale = measurement_settings.get("normalization_scale", 1)

    # efract and ifract have been populated in the species loops above

    # Call forward model to get the fit
    fit = _scattered_power_wavelength(
        n=n * 1e6,
        ue=ue,
        ui=ui,
        Te=Te * e / kB,
        Ti=Ti * e / kB,
        pe=pe,
        pi=pi,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_a=ion_a,
        wavelengths=wavelengths,
        probe_wavelength=probe_wavelength,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        ue_dir=ue_dir,
        ui_dir=ui_dir,
        instr_func_arr=instr_func_arr,
        normalization_type=normalization_type,
        normalization_scale=normalization_scale,
    )

    # Compute likelihood (data fidelity)
    ll = _log_likelihood(fit, Pkl_data, Pkl_var)

    # Compute prior penalty if requested
    prior = 0
    if use_penalty and penalty_settings is not None:
        prior = _log_prior(params, Nt, penalty_settings)

    return ll + prior


def run_fit(
    Pkl_data,
    Pkl_var,
    measurement_settings,
    penalty_settings=None,
    params_settings=None,
    fit_settings=None,
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

    Returns
    -------
    result : lmfit.MinimizerResult
        Full result from lmfit, including result.params with the best-fit
        values and result.success indicating convergence.
    """
    if fit_settings is None:
        fit_settings = {}
    fit_settings = dict(fit_settings)  # copy to avoid mutating caller's dict
    method = fit_settings.pop("method", "nelder")

    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    Nt = jnp.shape(Pkl_data)[1]

    params = build_params(Nelectrons, Nions, Nt, params_settings)

    Pkl_data = jnp.array(Pkl_data)
    Pkl_var = jnp.array(Pkl_var)

    def objective(p):
        return _log_posterior(p, Pkl_data, Pkl_var, measurement_settings, penalty_settings)

    minner = Minimizer(objective, params)
    result = minner.minimize(method=method, **fit_settings)

    return result


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
    Pkl_data, Pkl_var, measurement_settings, params_settings, fit_settings :
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
    """
    Pkl_data = jnp.array(Pkl_data)
    Pkl_var = jnp.array(Pkl_var)

    chi2_grid = jnp.zeros((len(weight_scales), len(cutoff_scales)))
    for i, ws in enumerate(weight_scales):
        for j, cs in enumerate(cutoff_scales):
            scaled = _scale_penalty_settings(penalty_settings, ws, cs)
            result = run_fit(
                Pkl_data, Pkl_var, measurement_settings,
                penalty_settings=scaled,
                params_settings=params_settings,
                fit_settings=fit_settings,
            )
            # Evaluate log likelihood only (no regularization) at best-fit params
            chi2_val = float(_log_posterior(
                result.params, Pkl_data, Pkl_var,
                measurement_settings, penalty_settings=None, use_penalty=False,
            ))
            chi2_grid = chi2_grid.at[i, j].set(chi2_val)

    return chi2_grid


def build_params(Nelectrons, Nions, Nt, params_settings=None):
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
        return {**default, **user}

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

    return p


