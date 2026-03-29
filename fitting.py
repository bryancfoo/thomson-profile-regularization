import jax.numpy as jnp
from lmfit import Parameters, Minimizer
from utility import extract_params_as_array
from forward import _scattered_power_wavelength

#Now for building the fitter

#Log likelihood of the fit being measured out of the data, obtained by averaging over the residuals
#The reason I average and not sum is to make the regularization weights not depend on number of timesteps
def log_likelihood(fit, data, variance):
    return jnp.mean((fit - data) ** 2 / variance)


#no input sanitization here
#param_profile should have shape (Nt, 1) while everything else should be
def tikhonov_penalty(param_array,
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
def log_prior(params, Nindices, penalty_settings):
    # penalty_settings should be a dict where each key is a param name
    total_penalty = 0
    for var_key in penalty_settings:
        # for each var, extract the relevant array and then compute the penalty
        # Note that the penalty_settings are customizable as dicts
        param_array = extract_params_as_array(params, var_key, Nindices)
        total_penalty += tikhonov_penalty(param_array, **penalty_settings[var_key])
    return total_penalty

#Now define the full objective function which sums the log_likelihood + log_prior to get the log posterior
def log_posterior(params, Skw_data, Skw_var, measurement_settings, penalty_settings, use_penalty = True):
    #First count the number of species for both electrons and ions
    Nelectrons = measurement_settings["Nelectrons"] #for the electrons I'll take this as an explicit input
    Nions = len(measurement_settings["ion_z"])
    Nt = jnp.shape(Skw_data)[1]
    ne = extract_params_as_array(params, "ne", Nt)
    Te = extract_params_as_array(params, "Te", Nt)
    Ti = extract_params_as_array(params, "Ti", Nt)
    ue = extract_params_as_array(params, "ue", Nt)
    ui = extract_params_as_array(params, "ui", Nt)
    pe = extract_params_as_array(params, "pe", Nt)
    pi = extract_params_as_array(params, "pi", Nt)


