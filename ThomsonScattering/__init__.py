# Enable double-precision JAX before any submodule (forward.py, fitting.py)
# imports jax.numpy.  The whole codebase assumes float64: parameter ranges
# span ~1e30 (densities), Tikhonov Hessians can be wildly ill-conditioned,
# and dual-averaging step sizes routinely shrink past float32's range
# (~1e-38).  Without this flag JAX silently truncates jnp.float64 to
# float32 and the resulting numerical noise is hard to diagnose.
import jax as _jax
_jax.config.update("jax_enable_x64", True)

from .deck import (
    load_deck, build_settings_from_deck,
    save_fit_results, save_posterior_samples,
)
from .fitting import run_fit_grad, build_params, compute_initial_fit, Param
from .forward import scattered_power_wavelength, spectral_density
from .sampling import build_sampling_problem, run_sgld_posterior

__all__ = [
    "load_deck", "build_settings_from_deck",
    "save_fit_results", "save_posterior_samples",
    "run_fit_grad", "build_params", "compute_initial_fit", "Param",
    "scattered_power_wavelength", "spectral_density",
    "build_sampling_problem", "run_sgld_posterior",
]
