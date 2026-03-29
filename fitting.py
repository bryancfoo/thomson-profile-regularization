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
    # penalty_settings should be a dict where each key is a param name
    total_penalty = 0
    for var_key in penalty_settings:
        # for each var, extract the relevant array and then compute the penalty
        # Note that the penalty_settings are customizable as dicts
        param_array = extract_params_as_array(params, var_key, Nindices)
        total_penalty += _tikhonov_penalty(param_array, **penalty_settings[var_key])
    return total_penalty

#Now define the full objective function which sums the log_likelihood + log_prior to get the log posterior
def _log_posterior(params, Skw_data, Skw_var, measurement_settings, penalty_settings, use_penalty = True):
    #First count the number of species for both electrons and ions
    Nelectrons = measurement_settings["Nelectrons"] #for the electrons I'll take this as an explicit input
    Nions = len(measurement_settings["ion_z"])
    Nt = jnp.shape(Skw_data)[1] #number of time steps is the second dimension of the data
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
    ll = _log_likelihood(fit, Skw_data, Skw_var)

    # Compute prior penalty if requested
    prior = 0
    if use_penalty and penalty_settings is not None:
        prior = _log_prior(params, Nt, penalty_settings)

    return ll + prior


def build_minimal_params(Nelectrons, Nions, Nt, initial_params=None):
    """Build an lmfit.Parameters object with the naming scheme used by the fitter.

    Parameters created follow the pattern <var><species>_<time>, e.g. `Te0_0`, `Ti1_3`.
    initial_params may be a dict providing initial guesses. Supported keys:
      - per-time, per-species: "ne1_0" -> used directly
      - per-species uniform: "ne1" -> applied to all times for species 1
      - global uniform: "ne" -> applied to all species and times

    The function also creates `efract{species}_{t}` and `ifract{species}_{t}` entries.
    Returns an lmfit.Parameters instance.
    """
    p = Parameters()
    if initial_params is None:
        initial_params = {}

    def _lookup(base, species, t, default):
        # check most-specific to least-specific
        key_specific = f"{base}{species}_{t}"
        key_species = f"{base}{species}"
        key_global = base
        if key_specific in initial_params:
            return initial_params[key_specific]
        if key_species in initial_params:
            return initial_params[key_species]
        if key_global in initial_params:
            return initial_params[key_global]
        return default
    
    # total electron density time-series `n_{t}` (no species index)
    for t in range(Nt):
        p.add(f"n_{t}", value=_lookup("n", 0, t, 1e20)) # default 1e20 m^-3, converted to cm^-3 in forward model

    # electron moments
    for s in range(Nelectrons):
        for t in range(Nt):
            p.add(f"Te{s}_{t}", value=_lookup("Te", s, t, 100.0))
            p.add(f"ue{s}_{t}", value=_lookup("ue", s, t, 0.0))
            p.add(f"pe{s}_{t}", value=_lookup("pe", s, t, 2.0))
            p.add(f"efract{s}_{t}", value=_lookup("efract", s, t, 1.0))

    # ion moments
    for s in range(Nions):
        for t in range(Nt):
            p.add(f"Ti{s}_{t}", value=_lookup("Ti", s, t, 100.0))
            p.add(f"ui{s}_{t}", value=_lookup("ui", s, t, 0.0))
            p.add(f"pi{s}_{t}", value=_lookup("pi", s, t, 2.0))
            p.add(f"ifract{s}_{t}", value=_lookup("ifract", s, t, 1.0))

    return p


