"""User-supplied distribution models for the custom-callable example.

Contract (see ThomsonScattering/distributions.py):
- first argument: normalized parallel velocity x = (v - u)/vth,
  vth = sqrt(2*T/m), passed as a JAX scalar (the package vmaps it);
- remaining arguments: shape parameters (their names become fit-parameter
  prefixes, suffixed with the species kind+index, e.g. ``fhote0``);
- returns the normalized 1D reduced distribution, ∫ g dx = 1;
- must be jnp-differentiable in x and in any free shape parameter.
"""
import jax.numpy as jnp


def two_temp(x, fhot=0.1, rhot=3.0):
    """Bi-Maxwellian: cold bulk at T plus a hot fraction at rhot*T.

    g(x) = (1-fhot)/sqrt(pi) * exp(-x^2)
         + fhot/sqrt(pi*rhot) * exp(-x^2/rhot)

    ``fhot`` is the hot-electron number fraction; ``rhot`` the hot/cold
    temperature ratio.
    """
    cold = (1.0 - fhot) * jnp.exp(-x ** 2) / jnp.sqrt(jnp.pi)
    hot = fhot * jnp.exp(-x ** 2 / rhot) / jnp.sqrt(jnp.pi * rhot)
    return cold + hot
