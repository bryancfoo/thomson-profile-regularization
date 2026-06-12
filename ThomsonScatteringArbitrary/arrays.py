"""Small array-shape helpers used by the forward model and fit code.

`reshape_moments` broadcasts a scalar / (Nt,) / (Nions, Nt) profile up to the
(Nions, Nt, 1) shape that the forward model expects.

`extract_params_as_array` pulls a (Nt,) array out of a parameter dict whose
keys follow the ``<prefix>_<t>`` naming convention used throughout the package.
"""
import jax.numpy as jnp


def reshape_moments(Q, Nions, Nt):
    # Forward model wants shape (Nions, Nt, 1) (the trailing axis broadcasts
    # over the wavelength grid). Accept scalars and (Nt,) / (Nions, Nt) inputs.
    if jnp.ndim(Q) == 0:
        return jnp.ones((Nions, Nt))[:, :, jnp.newaxis] * Q
    if jnp.ndim(Q) == 1:
        return Q[jnp.newaxis, :, jnp.newaxis]
    elif jnp.ndim(Q) == 2:
        return Q[:, :, jnp.newaxis]


def extract_params_as_array(params, var, Nindices):
    """Return ``[params[f"{var}_0"].value, ..., params[f"{var}_{N-1}"].value]``."""
    return jnp.asarray([params[f"{var}_{i}"].value for i in range(Nindices)])
