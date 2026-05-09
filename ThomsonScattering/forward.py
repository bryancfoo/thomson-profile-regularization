from typing import NamedTuple

import jax.numpy as jnp
from jax import vmap, jit
from scipy.constants import c, m_e, m_p
from . import plasma
from . import gain as _gain
from .dispersion import _Zprime
from jax.scipy.special import gamma, gammaincc
from jax.scipy.signal import convolve
from .utility import reshape_moments
import matplotlib.pyplot as plt


# Bundle of intermediates returned by `_spectral_density`. The susceptibilities
# and dielectric are kept on the same (Nk, Nt) grid as Skw so the gain
# correction can reuse them without recomputing _Zprime.
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
#Note: this takes lambda as input but is technically a function of omega (convention, I guess)
#backend function (for now)
#No input sanitization
#Everything should be in the shape [Nions, Nt, Nk]
def _spectral_density(
        n,
        ue,
        ui,
        Te,
        Ti,
        pe,
        pi,
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
        Nelectrons=1, #this input doesn't actually do anything, it's to allow the dict in the fitting functions to unpack more easily...
):
    #Compute the Thomson geometry
    scattering_angle = jnp.arccos(jnp.dot(probe_vec, scatter_vec))
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / jnp.linalg.norm(k_vec)

    #Compute thermal speeds of each species
    vTe = plasma.thermal_velocity(Te, m_e / m_p, coef = 2)
    vTi = plasma.thermal_velocity(Ti, ion_a, coef = 2)

    #Compute electron and ion densities of each population
    ne = n * efract
    #zbar = jnp.sum(ifract * ion_z, axis = 0)
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


    #Scattering parameter alpha
    #alpha = jnp.sqrt(2) * wpe / np.outer(k, vT_e)

    #Normalize the phase velocities to the thermal velocity
    zetae = we / (k * vTe)
    zetai = wi / (k * vTi)

    # Cache gamma evaluations on p — these were each computed at multiple call
    # sites, and now also avoid the Nk-fold redundancy that the dropped
    # jnp.repeat used to introduce inside _Zprime.
    g3_pe = gamma(3 / pe)
    g5_pe = gamma(5 / pe)
    g2_pe = gamma(2 / pe)
    ratio_pe = jnp.sqrt(2 / 3 * g5_pe / g3_pe)

    g3_pi = gamma(3 / pi)
    g5_pi = gamma(5 / pi)
    g2_pi = gamma(2 / pi)
    ratio_pi = jnp.sqrt(2 / 3 * g5_pi / g3_pi)

    #Also normalize to the characteristic velocity vp
    xe = zetae * ratio_pe
    xi = zetai * ratio_pi

    # Calculate the susceptibilities. _Zprime now broadcasts p against zeta
    # internally, so we pass pe / pi at their natural shape.
    # Use plasma_frequency_sq (no sqrt) so the gradient is finite when n=0.
    # sqrt(0) has an infinite gradient; squaring it back gives 0*inf = NaN in VJP.
    wpe_sq = plasma.plasma_frequency_sq(ne, 1, m_e / m_p)
    chiE = 2 * wpe_sq / (vTe * k)**2 * _Zprime(zetae, pe)

    wpi_sq = plasma.plasma_frequency_sq(ni, ion_z, ion_a)
    chiI = 2 * wpi_sq / (vTi * k) ** 2 * _Zprime(zetai, pi)

    #longitudinal dielectric function
    sum_chiE = jnp.sum(chiE, axis = 0)
    sum_chiI = jnp.sum(chiI, axis = 0)
    epsilon = 1 + sum_chiE + sum_chiI

    #electron and ion contributions to Skw
    econtr = efract * (
            2
            * jnp.pi
            / k
            / vTe
            / (2 * g3_pe)
            * ratio_pe
            * jnp.power(jnp.abs(1 - sum_chiE / epsilon), 2)
            * gammaincc(2 / pe, jnp.abs(xe) ** pe)
            * g2_pe
    )

    icontr = ifract * (
        2
        * jnp.pi
        * ion_z
        / k
        / vTi
        / (2 * g3_pi)
        * ratio_pi
        * jnp.power(jnp.abs(sum_chiE / epsilon), 2)
        * gammaincc(2 / pi, jnp.abs(xi) ** pi)
        * g2_pi
    )

    Skw = jnp.real(jnp.sum(econtr, axis = 0)+jnp.sum(icontr, axis = 0))

    # k, ks come in with a leading singleton (1, Nt, Nk); strip it so all
    # returned arrays share the (Nk, Nt) orientation of Skw.T.
    return SpectralDensityOut(
        Skw=Skw.T,
        sum_chiE=sum_chiE.T,
        sum_chiI=sum_chiI.T,
        epsilon=epsilon.T,
        k=k[0].T,
        ks=ks[0].T,
    )

# This is the user-facing function. It takes regular sized inputs and reshapes them as needed
# to be used in _spectral_density
# UNFINISHED
def spectral_density(
        n,
        ue,
        ui,
        Te,
        Ti,
        pe,
        pi,
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
        notch=None,
):
    Nelectrons = jnp.shape(efract)[0]
    Nions = jnp.shape(ifract)[0]
    Nt = jnp.shape(n)[0]

    #reshape everything to be (Nions, Nt, Nk)
    n = reshape_moments(n, Nions, Nt)
    ue = reshape_moments(ue, Nelectrons, Nt)
    ui = reshape_moments(ui, Nions, Nt)
    Te = reshape_moments(Te, Nelectrons, Nt)
    Ti = reshape_moments(Ti, Nions, Nt)
    pe = reshape_moments(pe, Nelectrons, Nt)
    pi = reshape_moments(pi, Nions, Nt)
    efract = reshape_moments(efract, Nelectrons, Nt)
    ifract = reshape_moments(ifract, Nions, Nt)
    ion_z = ion_z[:, jnp.newaxis, jnp.newaxis]
    ion_a = ion_a[:, jnp.newaxis, jnp.newaxis]
    wavelengths_3d = wavelengths[jnp.newaxis, jnp.newaxis, :]
    out = _spectral_density(
        n,
        ue,
        ui,
        Te,
        Ti,
        pe,
        pi,
        efract,
        ifract,
        ion_z,
        ion_a,
        wavelengths_3d,
        probe_wavelength,
        probe_vec,
        scatter_vec,
        ue_dir,
        ui_dir
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
def _scattered_power_wavelength(
        n,
        ue,
        ui,
        Te,
        Ti,
        pe,
        pi,
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
    Nelectrons = jnp.shape(efract)[0]
    Nions = jnp.shape(ifract)[0]
    Nt = jnp.shape(n)[0]

    #reshape everything to be (Nions, Nt, Nk)
    n = reshape_moments(n, Nions, Nt)
    ue = reshape_moments(ue, Nelectrons, Nt)
    ui = reshape_moments(ui, Nions, Nt)
    Te = reshape_moments(Te, Nelectrons, Nt)
    Ti = reshape_moments(Ti, Nions, Nt)
    pe = reshape_moments(pe, Nelectrons, Nt)
    pi = reshape_moments(pi, Nions, Nt)
    efract = reshape_moments(efract, Nelectrons, Nt)
    ifract = reshape_moments(ifract, Nions, Nt)
    ion_z = ion_z[:, jnp.newaxis, jnp.newaxis]
    ion_a = ion_a[:, jnp.newaxis, jnp.newaxis]

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
            n,
            ue,
            ui,
            Te,
            Ti,
            pe,
            pi,
            efract,
            ifract,
            ion_z,
            ion_a,
            wavelengths[jnp.newaxis, jnp.newaxis, :],
            probe_wavelength,
            probe_vec,
            svec,
            ue_dir,
            ui_dir,
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

    # normalization_type is a static arg under jit, so branching here is
    # compile-time: only the selected reduction is traced.
    if normalization_type == "max":
        norm = normalization_scale / jnp.nanmax(Pklam, axis=0)
    elif normalization_type == "sum":
        norm = normalization_scale / jnp.nansum(Pklam, axis=0)
    else:  # "integral"
        Pklam_finite = jnp.where(jnp.isnan(Pklam), 0.0, Pklam)
        norm = normalization_scale / jnp.trapezoid(Pklam_finite, wavelengths, axis=0)

    Pklam = Pklam * norm



    return Pklam


def scattered_power_wavelength(*args, **kwargs):

    return _scattered_power_wavelength(*args, **kwargs)
