"""Thomson-scattering forward model with per-species arbitrary distributions.

Same physics pipeline as the original ThomsonScattering package (geometry,
Doppler shifts, susceptibilities, dielectric screening, wavelength-space
conversion, IRF/throughput/notch/background/normalization, SRS/SBS gain), but
each electron/ion species carries a :class:`~.distributions.Distribution`
model instead of being hard-wired to a super-Gaussian:

    chi_s     = wp_s^2 / (vth_s * k)^2 * model_s.disp(zeta_s, shape_s)
    feature_s ∝ 2*pi / (k * vth_s) * |screening|^2 * model_s.reduced(zeta_s, shape_s)

For the analytic ``maxwellian`` / ``super_gaussian`` models this reproduces
the original code's tabulated-Z' + incomplete-gamma formulas exactly
(``disp = 2*_Zprime`` cancels the original's leading 2; ``reduced`` is the
identical bracket). General models route through the quadrature in
:mod:`.dispersion`.

Shape parameters are passed as ``e_shapes`` / ``i_shapes``: a tuple over
species of tuples of (Nt,) arrays, ordered as each model's
``shape_param_names``. The models themselves are static Python objects
(close over them or mark them static under jit).
"""
from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap
from scipy.constants import c, m_e, m_p
from . import plasma
from . import gain as _gain
from .arrays import reshape_moments


# Bundle of intermediates returned by `_spectral_density`. The susceptibilities
# and dielectric are kept on the same (Nk, Nt) grid as Skw so the gain
# correction can reuse them without recomputing the dispersion functions.
class SpectralDensityOut(NamedTuple):
    Skw: jnp.ndarray       # (Nk, Nt) real
    sum_chiE: jnp.ndarray  # (Nk, Nt) complex
    sum_chiI: jnp.ndarray  # (Nk, Nt) complex
    epsilon: jnp.ndarray   # (Nk, Nt) complex
    k: jnp.ndarray         # (Nk, Nt) real, fluctuation wavenumber
    ks: jnp.ndarray        # (Nk, Nt) real, scattered-light wavenumber


# Relevant normalization units are:
# Density: m^-3
# Velocity: m/s
# Temperature: eV
# Charge: e
# Mass: m_p


#Computes spectral density S(k, w)
#Backend function; no input sanitization.
#Moment arrays come in as (Nspecies, Nt, 1); wavelengths as (1, 1, Nk).
#e_shapes / i_shapes are tuples (per species) of tuples of (Nt,) arrays.
def _spectral_density(
        n,
        ue,
        ui,
        Te,
        Ti,
        e_shapes,
        i_shapes,
        efract,
        ifract,
        ion_z,
        ion_a,
        wavelengths,
        probe_wavelength,
        probe_vec,
        scatter_vec,
        ue_dir,
        ui_dir,
        e_models,
        i_models,
):
    Nelectrons = len(e_models)
    Nions = len(i_models)

    #Compute the Thomson geometry
    scattering_angle = jnp.arccos(jnp.dot(probe_vec, scatter_vec))
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / jnp.linalg.norm(k_vec)

    #Compute thermal speeds of each species
    vTe = plasma.thermal_velocity(Te, m_e / m_p, coef = 2)
    vTi = plasma.thermal_velocity(Ti, ion_a, coef = 2)

    #Compute electron and ion densities of each population
    ne = n * efract
    ni = n * ifract / ion_z # Note this is charge fraction not ion number fraction...

    #Compute total plasma frequency
    wpe_tot = plasma.plasma_frequency(n, 1, m_e / m_p)

    #Convert wavelengths to angular frequencies
    ws = 2 * jnp.pi * c / wavelengths
    wl = 2 * jnp.pi * c / probe_wavelength

    #Compute the frequency shift
    w = ws - wl

    #Compute wavenumbers
    ks = jnp.sqrt(ws ** 2 - wpe_tot ** 2) / c
    kl = jnp.sqrt(wl ** 2 - wpe_tot ** 2) / c
    k = jnp.sqrt(ks ** 2 + kl ** 2 - 2 * ks * kl * jnp.cos(scattering_angle))

    #Compute Doppler-shifted frequency w - k.u
    we = w - ue * k * jnp.dot(ue_dir, k_vec)
    wi = w - ui * k * jnp.dot(ui_dir, k_vec)

    #Normalize the phase velocities to the thermal velocity
    zetae = we / (k * vTe)   # (Nelectrons, Nt, Nk)
    zetai = wi / (k * vTi)   # (Nions, Nt, Nk)

    # Susceptibilities, species by species: each model supplies the full
    # generalized dispersion derivative Zgen = PV-Hilbert(g') + i*pi*g', so
    # chi = wp^2/(vth*k)^2 * Zgen (no leading 2 — for the analytic
    # super-Gaussian models, disp = 2*_Zprime absorbs the original factor).
    # Use plasma_frequency_sq (no sqrt) so the gradient is finite when n=0.
    wpe_sq = plasma.plasma_frequency_sq(ne, 1, m_e / m_p)
    wpi_sq = plasma.plasma_frequency_sq(ni, ion_z, ion_a)

    k2d = k[0]  # (Nt, Nk)

    chiE = [
        wpe_sq[s] / (vTe[s] * k2d) ** 2
        * e_models[s].disp(zetae[s], e_shapes[s])
        for s in range(Nelectrons)
    ]
    chiI = [
        wpi_sq[s] / (vTi[s] * k2d) ** 2
        * i_models[s].disp(zetai[s], i_shapes[s])
        for s in range(Nions)
    ]

    #longitudinal dielectric function
    sum_chiE = sum(chiE)
    sum_chiI = sum(chiI)
    epsilon = 1 + sum_chiE + sum_chiI

    #electron and ion contributions to Skw: the reduced 1D distribution
    #evaluated at the phase velocity, weighted by the dielectric screening
    e_screen = jnp.power(jnp.abs(1 - sum_chiE / epsilon), 2)
    i_screen = jnp.power(jnp.abs(sum_chiE / epsilon), 2)

    econtr = [
        efract[s] * 2 * jnp.pi / (k2d * vTe[s])
        * e_screen
        * e_models[s].reduced(zetae[s], e_shapes[s])
        for s in range(Nelectrons)
    ]
    icontr = [
        ifract[s] * 2 * jnp.pi * ion_z[s] / (k2d * vTi[s])
        * i_screen
        * i_models[s].reduced(zetai[s], i_shapes[s])
        for s in range(Nions)
    ]

    Skw = jnp.real(sum(econtr) + sum(icontr))

    # All returned arrays share the (Nk, Nt) orientation of Skw.T.
    return SpectralDensityOut(
        Skw=Skw.T,
        sum_chiE=sum_chiE.T,
        sum_chiI=sum_chiI.T,
        epsilon=epsilon.T,
        k=k[0].T,
        ks=ks[0].T,
    )


def _normalize_shapes(shapes, models, Nt, kind):
    """Coerce per-species shape params into tuples of (Nt,) float arrays."""
    if shapes is None:
        shapes = tuple(() for _ in models)
    if len(shapes) != len(models):
        raise ValueError(
            f"{kind}_shapes has {len(shapes)} species entries; expected "
            f"{len(models)} (one per model)."
        )
    out = []
    for s, (model, sh) in enumerate(zip(models, shapes)):
        sh = tuple(jnp.broadcast_to(jnp.asarray(v, dtype=jnp.float64), (Nt,))
                   for v in sh)
        if len(sh) != len(model.shape_param_names):
            raise ValueError(
                f"{kind}_shapes[{s}] has {len(sh)} entries; model "
                f"{model.name!r} expects {model.shape_param_names}."
            )
        out.append(sh)
    return tuple(out)


# This is the user-facing spectral density. It takes regular sized inputs and
# reshapes them as needed for _spectral_density.
def spectral_density(
        n,
        ue,
        ui,
        Te,
        Ti,
        efract,
        ifract,
        ion_z,
        ion_a,
        wavelengths,
        probe_wavelength,
        probe_vec,
        scatter_vec,
        ue_dir,
        ui_dir,
        e_models,
        i_models,
        e_shapes=None,
        i_shapes=None,
        notch=None,
):
    Nelectrons = len(e_models)
    Nions = len(i_models)
    Nt = jnp.shape(n)[0]

    #reshape everything to be (Nspecies, Nt, 1)
    n = reshape_moments(n, Nions, Nt)
    ue = reshape_moments(ue, Nelectrons, Nt)
    ui = reshape_moments(ui, Nions, Nt)
    Te = reshape_moments(Te, Nelectrons, Nt)
    Ti = reshape_moments(Ti, Nions, Nt)
    efract = reshape_moments(efract, Nelectrons, Nt)
    ifract = reshape_moments(ifract, Nions, Nt)
    ion_z = jnp.asarray(ion_z)[:, jnp.newaxis, jnp.newaxis]
    ion_a = jnp.asarray(ion_a)[:, jnp.newaxis, jnp.newaxis]
    e_shapes = _normalize_shapes(e_shapes, e_models, Nt, "e")
    i_shapes = _normalize_shapes(i_shapes, i_models, Nt, "i")

    out = _spectral_density(
        n, ue, ui, Te, Ti, e_shapes, i_shapes, efract, ifract,
        ion_z, ion_a,
        wavelengths[jnp.newaxis, jnp.newaxis, :],
        probe_wavelength, probe_vec, scatter_vec, ue_dir, ui_dir,
        e_models, i_models,
    )
    Skw = out.Skw

    # Apply notch: NaN out wavelengths between notch[0] and notch[1]
    if notch is not None:
        mask = (wavelengths >= notch[0]) & (wavelengths <= notch[1])
        Skw = jnp.where(mask[:, jnp.newaxis], jnp.nan, Skw)

    return Skw


#Computes the wavelength spectrum (NOT the frequency spectrum!) of the scattered power
#This is what you download off omegaops
#Normalization options might be helpful for data analysis
def scattered_power_wavelength(
        n,
        ue,
        ui,
        Te,
        Ti,
        efract,
        ifract,
        ion_z,
        ion_a,
        wavelengths,
        probe_wavelength,
        probe_vec,
        scatter_vec,
        ue_dir,
        ui_dir,
        e_models,
        i_models,
        e_shapes = None,
        i_shapes = None,
        instr_func_arr = None,
        irf_normalization = "area",
        throughput = None,
        aperture_weights = None,
        background_coefs = None,
        normalization_type = "max",
        normalization_scale = 1,
        notch = None,
        probe_intensity = 0.0,
        probe_diameter = 1.0,
        pol_p_fraction = 1.0,
        gain_mode = "off",
):
    Nelectrons = len(e_models)
    Nions = len(i_models)
    Nt = jnp.shape(n)[0]

    #reshape everything to be (Nspecies, Nt, 1)
    n = reshape_moments(n, Nions, Nt)
    ue = reshape_moments(ue, Nelectrons, Nt)
    ui = reshape_moments(ui, Nions, Nt)
    Te = reshape_moments(Te, Nelectrons, Nt)
    Ti = reshape_moments(Ti, Nions, Nt)
    efract = reshape_moments(efract, Nelectrons, Nt)
    ifract = reshape_moments(ifract, Nions, Nt)
    ion_z = jnp.asarray(ion_z)[:, jnp.newaxis, jnp.newaxis]
    ion_a = jnp.asarray(ion_a)[:, jnp.newaxis, jnp.newaxis]
    e_shapes = _normalize_shapes(e_shapes, e_models, Nt, "e")
    i_shapes = _normalize_shapes(i_shapes, i_models, Nt, "i")

    # Promote scatter_vec to (Nangles, 3); a single (3,) becomes (1, 3).
    # Aperture-averaged spectrum: vmap _spectral_density over the angle axis,
    # then weighted-sum with aperture_weights.
    scatter_vec_arr = jnp.atleast_2d(scatter_vec)
    if aperture_weights is None:
        weights = jnp.ones(scatter_vec_arr.shape[0]) / scatter_vec_arr.shape[0]
    else:
        weights = jnp.asarray(aperture_weights)

    def _skw_one(svec):
        out = _spectral_density(
            n, ue, ui, Te, Ti, e_shapes, i_shapes, efract, ifract,
            ion_z, ion_a,
            wavelengths[jnp.newaxis, jnp.newaxis, :],
            probe_wavelength, probe_vec, svec, ue_dir, ui_dir,
            e_models, i_models,
        )
        # Apply Turnbull et al. (PRL 2026) SRS/SBS gain correction per ray.
        # Each aperture-averaged scatter angle sees its own theta_s, so L and
        # the cos^2 polarization factor are computed inside this vmap branch.
        scattering_angle = jnp.arccos(jnp.dot(probe_vec, svec))
        G = _gain.gain_factor(
            out.sum_chiE, out.sum_chiI, out.epsilon, out.k, out.ks,
            scattering_angle=scattering_angle,
            probe_wavelength=probe_wavelength,
            probe_intensity=probe_intensity,
            probe_diameter=probe_diameter,
            pol_p_fraction=pol_p_fraction,
            mode=gain_mode,
        )
        return out.Skw * G

    Skw_stack = vmap(_skw_one)(scatter_vec_arr)  # (Nangles, Nk, Nt)
    Skw = jnp.tensordot(weights, Skw_stack, axes=1)  # (Nk, Nt)

    #Convert to wavelength space
    #Correction by dw/d(lambda) ~ lambda**(-2)
    Sklam = Skw / wavelengths[:, jnp.newaxis]**2

    #Now correct by (1+2w/wl) as given in Sheffield Eq. 5.1
    ws = 2 * jnp.pi * c / wavelengths
    wl = 2 * jnp.pi * c / probe_wavelength
    w = ws - wl

    w = w[:, jnp.newaxis]

    Pklam = Sklam * (1 + 2 * w / wl)

    # Apply wavelength-dependent throughput (spectrometer transmission/sensitivity)
    # before the IRF: throughput modulates the true signal, then the IRF smears it.
    if throughput is not None:
        Pklam = Pklam * throughput[:, jnp.newaxis]

    # Here I assume that the instrument function is applied to the scattered power
    # and not to Skw, which I think is what the file I get from Joe Katz does...
    if instr_func_arr is not None:
        # Assuming a time-dependent instrument function, we use jax.vmap to apply the
        # relevant convolution to each time step
        # Not using 2D convolution to avoid time smearing
        Pklam = vmap(lambda p, i: jnp.convolve(p, i, mode="same"), in_axes=1, out_axes=1)(Pklam, instr_func_arr)
        # Renormalize so amplitude isn't coupled to PSF area / peak.
        if irf_normalization == "area":
            Pklam = Pklam / jnp.sum(instr_func_arr, axis=0, keepdims=True)
        elif irf_normalization == "peak":
            Pklam = Pklam / jnp.max(instr_func_arr, axis=0, keepdims=True)
        # "none" leaves Pklam unscaled.

    # Apply notch: NaN out wavelengths between notch[0] and notch[1]
    if notch is not None:
        mask = (wavelengths >= notch[0]) & (wavelengths <= notch[1])
        Pklam = jnp.where(mask[:, jnp.newaxis], jnp.nan, Pklam)

    # Polynomial background in centered+scaled wavelength.
    # background_coefs has shape (K+1, Nt); coef i is the (lam-lam0)/lam0 ** i term.
    # Added after notch (so the notched region stays NaN) and before normalization
    # (so signal+bg get normalized together, matching how data is processed).
    if background_coefs is not None:
        K_plus_1 = background_coefs.shape[0]
        lam_norm = (wavelengths - probe_wavelength) / probe_wavelength
        i_arr = jnp.arange(K_plus_1)
        powers = lam_norm[:, jnp.newaxis] ** i_arr[jnp.newaxis, :]  # (Nk, K+1)
        Pklam = Pklam + powers @ background_coefs  # (Nk, Nt)

    # Replace NaN (notch pixels) before multiplying by norm so the VJP of
    # (Pklam * norm) never computes 0 * NaN = NaN when backpropagating through
    # the shared norm scalar.  _log_likelihood masks these pixels via isnan(data).
    Pklam_finite = jnp.where(jnp.isnan(Pklam), 0.0, Pklam)

    # normalization_type is a static arg under jit, so branching here is
    # compile-time: only the selected reduction is traced.
    if normalization_type == "max":
        norm = normalization_scale / jnp.nanmax(Pklam_finite, axis=0)
    elif normalization_type == "sum":
        norm = normalization_scale / jnp.nansum(Pklam_finite, axis=0)
    else:  # "integral"
        norm = normalization_scale / jnp.trapezoid(Pklam_finite, wavelengths, axis=0)

    Pklam = Pklam_finite * norm

    return Pklam
