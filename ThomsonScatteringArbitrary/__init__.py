# Expose the host CPU as N XLA devices for intra-fit time-axis sharding, when
# requested via the THOMSON_CPU_DEVICES env var. This MUST happen before the
# first `import jax` (XLA reads the device-count flag at initialization), which
# is why it lives here at the top of the package __init__. The CLI sets the env
# var from --n-devices before importing this package; library/notebook users
# can export it themselves.
import os as _os


def _apply_cpu_device_count():
    # CLI flags must be honored here, not in the CLI modules: under
    # `python -m ThomsonScatteringArbitrary.thomson_fit`, this __init__ (and
    # therefore the first jax import) runs before the submodule's own code.
    import sys as _sys
    argv = _sys.argv
    if "--serial" in argv:
        _os.environ["THOMSON_NO_PARALLEL"] = "1"
    # Global kill-switch: THOMSON_NO_PARALLEL forces single-device (no sharding).
    if _os.environ.get("THOMSON_NO_PARALLEL", "").strip().lower() not in (
            "", "0", "false", "no"):
        return
    n = _os.environ.get("THOMSON_CPU_DEVICES")
    for _i, _a in enumerate(argv):
        if _a == "--n-devices" and _i + 1 < len(argv):
            n = argv[_i + 1]
        elif _a.startswith("--n-devices="):
            n = _a.split("=", 1)[1]
    if n:
        _os.environ["THOMSON_CPU_DEVICES"] = str(n)
    if not n:
        return
    try:
        n = int(n)
    except (TypeError, ValueError):
        return
    if n <= 1:
        return
    flag = "--xla_force_host_platform_device_count"
    cur = _os.environ.get("XLA_FLAGS", "")
    if flag not in cur:
        _os.environ["XLA_FLAGS"] = (cur + f" {flag}={n}").strip()


_apply_cpu_device_count()

# Enable double-precision JAX before any submodule (forward.py, fitting.py)
# imports jax.numpy.  The whole codebase assumes float64: parameter ranges
# span ~1e30 (densities), Tikhonov Hessians can be wildly ill-conditioned,
# and dual-averaging step sizes routinely shrink past float32's range
# (~1e-38).  Without this flag JAX silently truncates jnp.float64 to
# float32 and the resulting numerical noise is hard to diagnose.
import jax as _jax
_jax.config.update("jax_enable_x64", True)

from .deck import (
    load_deck, build_settings_from_deck, save_fit_results,
)
from .distributions import (
    Distribution, GeneralDistribution, Maxwellian, SuperGaussian,
    resolve_distribution, resolve_models,
)
from .fitting import run_fit_grad, build_params, compute_initial_fit, Param
from .forward import scattered_power_wavelength, spectral_density
from .sampling import build_sampling_problem, run_sgld_posterior
from .l_curve import compute_L_curve

__all__ = [
    "load_deck", "build_settings_from_deck", "save_fit_results",
    "Distribution", "GeneralDistribution", "Maxwellian", "SuperGaussian",
    "resolve_distribution", "resolve_models",
    "run_fit_grad", "build_params", "compute_initial_fit", "Param",
    "scattered_power_wavelength", "spectral_density",
    "build_sampling_problem", "run_sgld_posterior",
    "compute_L_curve",
]
