"""Per-species velocity-distribution models for the Thomson forward model.

Every species in the forward model carries a :class:`Distribution` exposing
two JAX functions of the normalized phase velocity zeta = (w - k·u)/(k·vth),
vth = sqrt(2·T/m):

- ``disp(zeta, shape)``    → complex generalized dispersion derivative
      Zgen(zeta) = P∫ g'(x)/(zeta − x) dx + i·pi·g'(zeta)
  (the package's Sheffield sign convention, equal to ``2·_Zprime`` for
  super-Gaussians), so the susceptibility is
      chi = wp² / (vth·k)² · disp(zeta).
- ``reduced(zeta, shape)`` → the normalized 1D reduced distribution g(zeta)
  (∫ g dx = 1), which enters the spectral-density feature terms.

``shape`` is a tuple of per-time arrays (one per entry of
``shape_param_names``, each broadcastable against zeta's time axis).

Velocity convention
-------------------
``g`` is the 1D distribution *reduced along the scattering wavevector k*,
expressed in x = (v∥ − u)/vth with vth = sqrt(2·T/m). A Maxwellian is
g(x) = exp(−x²)/sqrt(pi) (variance 1/2). The temperature parameter Te/Ti in
the fit sets vth; whether it equals the thermodynamic temperature for a
non-Maxwellian family depends on that family's own convention — document it
in the model.

Two kinds of models:

- *Analytic* (``maxwellian``, ``super_gaussian``): closed-form g plus the
  tabulated ``_Zprime`` from :mod:`.dispersion` (``disp = 2·_Zprime``,
  numerically identical to the original ThomsonScattering package).
- *General* (everything else, including user callables): g supplied as a JAX
  scalar function ``g(x, *shape)``; derivatives via ``jax.grad`` and the
  dispersion integral via singularity-subtraction quadrature
  (:func:`.dispersion.hilbert_disp`) on a fixed normalized grid.

Deck-facing model specs are plain strings/dicts (picklable across the
L-curve process pool); :func:`resolve_distribution` turns one into a
Distribution object inside each process.
"""
import importlib.util
import inspect
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, vmap
from jax.scipy.special import gamma, gammaincc, gammaln

from .dispersion import _Zprime, hilbert_disp, simpson_grid


# ── parameter-name plumbing ──────────────────────────────────────────────────

# Shape-parameter prefixes are built as f"{name}{kind}{species_idx}" with kind
# in {"e", "i"} — e.g. the super-Gaussian exponent "p" on electron species 0
# becomes "pe0", matching the original package's naming exactly. Names must
# not contain underscores (the deck parser splits per-time keys on "_") and
# must not collide with the universal moment bases once suffixed.
_RESERVED_SUFFIXED = {"Te", "ue", "efract", "Ti", "ui", "ifract", "ne", "ni"}


def shape_param_prefix(name, kind, species_idx):
    """Deck/param prefix for shape param ``name`` of species ``species_idx``."""
    return f"{name}{kind}{species_idx}"


def _validate_shape_names(names, model_name):
    for nm in names:
        if "_" in nm:
            raise ValueError(
                f"Distribution {model_name!r}: shape parameter {nm!r} contains "
                "'_', which conflicts with the '<prefix>_<time>' parameter "
                "naming. Rename it without underscores."
            )
        for kind in ("e", "i"):
            if f"{nm}{kind}" in _RESERVED_SUFFIXED:
                raise ValueError(
                    f"Distribution {model_name!r}: shape parameter {nm!r} "
                    f"would produce prefix base {nm + kind!r}, which collides "
                    "with a reserved moment parameter name."
                )


# ── base class ───────────────────────────────────────────────────────────────

class Distribution:
    """Base class; see module docstring for the disp/reduced contract."""

    #: model name (registry key or "path.py:func")
    name = "?"
    #: shape parameter names, in the order ``shape`` tuples are passed
    shape_param_names = ()
    #: default Param settings per shape param: {name: {"value": ..., ...}}
    shape_param_defaults = {}

    def disp(self, zeta, shape):
        raise NotImplementedError

    def reduced(self, zeta, shape):
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__} {self.name!r} shape={self.shape_param_names}>"


# ── analytic models (tabulated _Zprime fast path) ────────────────────────────

def _supergauss_reduced(zeta, p):
    """Normalized 1D reduction of the isotropic 3D super-Gaussian of order p.

    Identical to the feature term of the original forward model:
    ratio_p/(2·Γ(3/p)) · Γ(2/p) · gammaincc(2/p, |ratio_p·zeta|^p),
    ratio_p = sqrt(2/3 · Γ(5/p)/Γ(3/p)). For p = 2 this is exp(−zeta²)/sqrt(pi).
    """
    g3 = gamma(3 / p)
    g5 = gamma(5 / p)
    g2 = gamma(2 / p)
    ratio = jnp.sqrt(2 / 3 * g5 / g3)
    x = zeta * ratio
    return ratio / (2 * g3) * g2 * gammaincc(2 / p, jnp.abs(x) ** p)


class Maxwellian(Distribution):
    """Maxwellian: g(x) = exp(−x²)/sqrt(pi). No shape parameters.

    ``disp`` reuses the tabulated super-Gaussian Z' at p = 2 — exactly what the
    original package computes for a Maxwellian — so results are bit-compatible.
    """
    name = "maxwellian"
    shape_param_names = ()
    shape_param_defaults = {}

    def disp(self, zeta, shape):
        return 2.0 * _Zprime(zeta, jnp.full((1,) * jnp.ndim(zeta), 2.0))

    def reduced(self, zeta, shape):
        return jnp.exp(-zeta ** 2) / jnp.sqrt(jnp.pi)


class SuperGaussian(Distribution):
    """Isotropic super-Gaussian of order p ∈ [2, 5] (tabulated fast path)."""
    name = "super_gaussian"
    shape_param_names = ("p",)
    shape_param_defaults = {"p": {"value": 2.0, "min": 2.0, "max": 5.0}}

    def disp(self, zeta, shape):
        (p,) = shape
        return 2.0 * _Zprime(zeta, _broadcast_shape_arr(p, zeta))

    def reduced(self, zeta, shape):
        (p,) = shape
        return _supergauss_reduced(zeta, _broadcast_shape_arr(p, zeta))


def _broadcast_shape_arr(s, zeta):
    """Promote a (Nt,)-shaped (or scalar) shape param against (..., Nt, Nk) zeta."""
    s = jnp.asarray(s)
    if s.ndim == 0:
        return s[jnp.newaxis]
    return s[..., :, jnp.newaxis]  # (..., Nt) → (..., Nt, 1)


# ── general models (quadrature path) ─────────────────────────────────────────

class GeneralDistribution(Distribution):
    """Arbitrary distribution from a JAX scalar callable ``g(x, *shape)``.

    ``g`` must return the normalized 1D reduced distribution (∫ g dx = 1) at
    scalar x; it is evaluated through ``vmap`` so it never needs to handle
    array broadcasting itself, and must be differentiable (``jax.grad``) in x
    and in any shape parameter that is free in the fit.

    The dispersion integral runs on a fixed Simpson grid over
    [−x_max, x_max]; choose ``x_max`` large enough that g' is negligible
    outside (raise it for fat-tailed families like kappa).
    """

    def __init__(self, g, shape_param_names, name=None,
                 shape_param_defaults=None, x_max=10.0, n_points=2001):
        self.g = g
        self.shape_param_names = tuple(shape_param_names)
        self.name = name or getattr(g, "__name__", "custom")
        self.shape_param_defaults = dict(shape_param_defaults or {})
        _validate_shape_names(self.shape_param_names, self.name)
        self.x_max = float(x_max)
        self.n_points = int(n_points)
        self.x_grid, self.weights = simpson_grid(self.x_max, self.n_points)

        n_shape = len(self.shape_param_names)
        in_axes = (0,) + (None,) * n_shape
        self._g_v = vmap(g, in_axes=in_axes)
        self._gp_v = vmap(jax.grad(g, argnums=0), in_axes=in_axes)
        self._gpp_v = vmap(jax.grad(jax.grad(g, argnums=0), argnums=0),
                           in_axes=in_axes)

    def _map_time(self, fn, zeta, shape):
        """Apply ``fn(zeta_row, shape_scalars)`` over the time axis.

        zeta is (Nt, Nk) and each shape entry is (Nt,) (or scalar). Time is
        mapped sequentially with ``lax.map`` so the (Nk, Nx) quadrature slab
        is the peak memory, not (Nt, Nk, Nx). Any extra leading batch dims
        (e.g. the aperture vmap) are handled by JAX's batching rules without
        this code seeing them.
        """
        shape = tuple(jnp.broadcast_to(jnp.asarray(s), zeta.shape[:-1])
                      for s in shape)
        if zeta.ndim == 1:
            return fn(zeta, shape)

        def _one(args):
            z_row, s_row = args
            return fn(z_row, s_row)
        return lax.map(_one, (zeta, shape))

    def disp(self, zeta, shape):
        def _disp_row(z_row, s_row):
            gp_grid = self._gp_v(self.x_grid, *s_row)
            gp_z = self._gp_v(z_row, *s_row)
            gpp_z = self._gpp_v(z_row, *s_row)
            return hilbert_disp(z_row, gp_z, gpp_z, gp_grid,
                                self.x_grid, self.weights)
        return self._map_time(_disp_row, jnp.asarray(zeta), shape)

    def reduced(self, zeta, shape):
        def _red_row(z_row, s_row):
            return self._g_v(z_row, *s_row)
        return self._map_time(_red_row, jnp.asarray(zeta), shape)

    def check_normalization(self, shape_values, atol=1e-3):
        """Eager sanity check: ∫ g dx on the quadrature grid vs 1.

        ``shape_values`` is a tuple of representative scalars (e.g. deck
        initial values). Returns the integral; caller decides whether to warn.
        """
        g_vals = np.asarray(self._g_v(self.x_grid,
                                      *[jnp.asarray(float(v)) for v in shape_values]))
        return float(np.sum(np.asarray(self.weights) * g_vals))


# ── built-in general families ────────────────────────────────────────────────

def _kappa_g(x, kappa):
    """1D kappa (Lorentzian-tailed) distribution in x = v/vth units:

    g(x; κ) = Γ(κ)/(sqrt(pi·κ)·Γ(κ−1/2)) · (1 + x²/κ)^(−κ),  κ > 3/2.

    Convention: the kappa "thermal speed" is taken equal to vth = sqrt(2T/m),
    i.e. T is the kappa-temperature of the family, not the variance
    temperature (the variance is κ/(2κ−3) · 1/2 · 2 = κ/(2κ−3), → 1/2 as
    κ → ∞ where the Maxwellian is recovered).
    """
    norm = jnp.exp(gammaln(kappa) - gammaln(kappa - 0.5)) / jnp.sqrt(jnp.pi * kappa)
    return norm * (1.0 + x ** 2 / kappa) ** (-kappa)


def _supergauss_numeric_g(x, p):
    """The projected super-Gaussian as a general-path callable (validation)."""
    return _supergauss_reduced(x, p)


# Registry: name → factory(options_dict) → Distribution. Options come from the
# deck spec (e.g. x_max / n_points overrides for the general path).
def _make_maxwellian(opts):
    return Maxwellian()


def _make_super_gaussian(opts):
    return SuperGaussian()


def _make_kappa(opts):
    return GeneralDistribution(
        _kappa_g, ("kappa",), name="kappa",
        shape_param_defaults={"kappa": {"value": 4.0, "min": 1.6, "max": 50.0}},
        x_max=opts.get("x_max", 20.0),
        n_points=opts.get("n_points", 4001),
    )


def _make_super_gaussian_numeric(opts):
    return GeneralDistribution(
        _supergauss_numeric_g, ("p",), name="super_gaussian_numeric",
        shape_param_defaults={"p": {"value": 2.0, "min": 2.0, "max": 5.0}},
        x_max=opts.get("x_max", 10.0),
        n_points=opts.get("n_points", 2001),
    )


_REGISTRY = {
    "maxwellian": _make_maxwellian,
    "super_gaussian": _make_super_gaussian,
    "kappa": _make_kappa,
    "super_gaussian_numeric": _make_super_gaussian_numeric,
}


# ── custom-callable loading ──────────────────────────────────────────────────

def _load_callable(path_spec, base_dir=None):
    """Load ``"file.py:func"`` (path resolved against base_dir) → callable."""
    file_part, func_name = path_spec.rsplit(":", 1)
    file_path = pathlib.Path(file_part)
    if not file_path.is_absolute() and base_dir is not None:
        file_path = pathlib.Path(base_dir) / file_path
    file_path = file_path.resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"Distribution module not found: {file_path}")
    mod_name = f"_thomson_dist_{file_path.stem}_{abs(hash(str(file_path))) % 10**8}"
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
    else:
        spec = importlib.util.spec_from_file_location(mod_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
    try:
        fn = getattr(mod, func_name)
    except AttributeError:
        raise AttributeError(
            f"{file_path} has no function {func_name!r}."
        ) from None
    return fn, str(file_path)


def _introspect_shape_params(fn):
    """Shape param names + defaults from the callable signature (args after x)."""
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if len(params) < 1:
        raise ValueError(
            f"Distribution callable {fn.__name__!r} must take the normalized "
            "velocity x as its first argument."
        )
    names, defaults = [], {}
    for prm in params[1:]:
        if prm.kind in (inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD):
            raise ValueError(
                f"Distribution callable {fn.__name__!r}: *args/**kwargs are not "
                "supported; declare each shape parameter explicitly."
            )
        names.append(prm.name)
        if prm.default is not inspect.Parameter.empty:
            defaults[prm.name] = {"value": float(prm.default)}
    return tuple(names), defaults


# ── spec resolution ──────────────────────────────────────────────────────────

def resolve_distribution(spec, base_dir=None):
    """Turn a model spec into a Distribution.

    ``spec`` forms:
    - Distribution instance               → returned as-is
    - ``"maxwellian"`` etc.               → registry model
    - ``"file.py:func"``                  → custom callable (general path)
    - dict with key ``"model"`` plus options (``x_max``, ``n_points``)
    """
    if isinstance(spec, Distribution):
        return spec
    opts = {}
    if isinstance(spec, dict):
        opts = {k: v for k, v in spec.items() if k != "model"}
        spec = spec["model"]
    if not isinstance(spec, str):
        raise TypeError(f"Cannot resolve distribution spec {spec!r}")
    if spec in _REGISTRY:
        return _REGISTRY[spec](opts)
    if ":" in spec and ".py" in spec:
        fn, abs_path = _load_callable(spec, base_dir=opts.pop("base_dir", base_dir))
        names, defaults = _introspect_shape_params(fn)
        return GeneralDistribution(
            fn, names, name=f"{abs_path}:{fn.__name__}",
            shape_param_defaults=defaults,
            x_max=opts.get("x_max", 10.0),
            n_points=opts.get("n_points", 2001),
        )
    raise ValueError(
        f"Unknown distribution model {spec!r}. Use one of "
        f"{sorted(_REGISTRY)} or a 'file.py:function' reference."
    )


def resolve_models(measurement_settings):
    """Resolve per-species model specs from a measurement_settings dict.

    Reads optional keys ``e_models`` / ``i_models`` (lists of specs, one per
    species; default ``"super_gaussian"`` for every species — backward
    compatible with the original package) and ``_model_base_dir`` for
    resolving relative custom-callable paths.

    Returns ``(e_models, i_models)`` as tuples of Distribution objects.
    """
    Nelectrons = measurement_settings["Nelectrons"]
    Nions = len(measurement_settings["ion_z"])
    base_dir = measurement_settings.get("_model_base_dir", None)

    e_specs = measurement_settings.get("e_models") or ["super_gaussian"] * Nelectrons
    i_specs = measurement_settings.get("i_models") or ["super_gaussian"] * Nions
    if len(e_specs) != Nelectrons:
        raise ValueError(
            f"e_models has {len(e_specs)} entries but Nelectrons = {Nelectrons}."
        )
    if len(i_specs) != Nions:
        raise ValueError(
            f"i_models has {len(i_specs)} entries but len(ion_z) = {Nions}."
        )
    e_models = tuple(resolve_distribution(s, base_dir) for s in e_specs)
    i_models = tuple(resolve_distribution(s, base_dir) for s in i_specs)
    return e_models, i_models
