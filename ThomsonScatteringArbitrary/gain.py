"""SRS / SBS convective-gain correction for Thomson scattering spectra.

Implements the "Thomson Scattering with Gain" correction from Turnbull et al.,
PRL 136, 135101 (2026). The scattered light, seeded by thermal Thomson noise
inside the probe-beam volume, is amplified or depleted by SRS (electron-plasma
wave) / SBS (ion-acoustic wave) instabilities before exiting. The net effect
multiplies S(k, omega) by

    G(k, omega) = (e^(gamma * L) - 1) / (gamma * L)        (Eq. 7)

with the spatial intensity-gain rate

    gamma = k^2 / (4 k_s) * Im(F_chi) * |a_tilde_0|^2      (Eq. 6)
    F_chi = chi_e * (1 + sum_j chi_{i,j}) / epsilon
    |a_tilde_0|^2 = |a_{0,p}|^2 cos^2(theta_s) + |a_{0,s}|^2

For small |gamma * L| the formula reduces to S * exp(gamma L / 2) (Eq. 8).
The exact form is preferred because |gamma L| reaches ~0.6 in real shots; the
small-gain branch is exposed as a sanity-check toggle.
"""

import jax.numpy as jnp


# Linear-pol normalized vector potential coefficient:
#     |a_0|^2 = _A0_SQ_COEFF * I[W/cm^2] * lambda^2[um^2]
# Derived from |a_0|^2 = e^2 I lambda^2 / (2 pi^2 m_e^2 c^5 eps_0) using
# I in W/m^2, lambda in m; converted to W/cm^2 and um.
_A0_SQ_COEFF = 7.3e-19


def gain_factor(
    sum_chiE, sum_chiI, epsilon, k, ks,
    scattering_angle, probe_wavelength,
    probe_intensity, probe_diameter,
    pol_p_fraction,
    mode,
):
    """Return the multiplicative gain correction G(k, omega).

    Parameters
    ----------
    sum_chiE, sum_chiI, epsilon : complex array, shape (Nk, Nt)
        Total electron susceptibility, total ion susceptibility, and
        longitudinal dielectric function from `_spectral_density`.
    k, ks : real array, shape (Nk, Nt)
        Fluctuation and scattered-light wavenumbers (rad / m).
    scattering_angle : scalar
        Angle between probe and scatter direction (rad).
    probe_wavelength : scalar
        Probe wavelength in meters.
    probe_intensity : scalar
        Probe intensity in W / cm^2. Set to 0 to disable amplification
        without changing the call structure.
    probe_diameter : scalar
        Probe-beam FWHM diameter in meters. Sets the gain length
        L = D_0 / sin(theta_s).
    pol_p_fraction : scalar in [0, 1]
        Fraction of probe power in the p-polarization (in scattering plane).
        1.0 = pure p-pol, 0.0 = pure s-pol.
    mode : {"exact", "small_gain", "off"}
        "exact"      : (e^(gL) - 1) / (gL) with safe gL -> 0 Taylor branch.
        "small_gain" : exp(gL / 2).
        "off"        : returns 1.0 (no correction).

    Returns
    -------
    G : array, shape (Nk, Nt)
        Multiplicative factor; multiply Skw by this before instrument response.
    """
    if mode == "off":
        return jnp.array(1.0)

    F_chi = sum_chiE * (1.0 + sum_chiI) / epsilon
    lam_um = probe_wavelength * 1e6
    a0_sq = _A0_SQ_COEFF * probe_intensity * lam_um ** 2
    cos2 = jnp.cos(scattering_angle) ** 2
    a_tilde_sq = a0_sq * (pol_p_fraction * cos2 + (1.0 - pol_p_fraction))

    gamma = (k ** 2 / (4.0 * ks)) * jnp.imag(F_chi) * a_tilde_sq
    L = probe_diameter / jnp.sin(scattering_angle)
    gL = gamma * L

    if mode == "small_gain":
        return jnp.exp(gL / 2.0)

    # Eq. 7 with safe gL -> 0 limit. expm1(x)/x is well-behaved at large |x|
    # but loses precision near zero; switch to a 4-term Taylor for |gL| < 1e-4.
    small = jnp.abs(gL) < 1e-4
    safe_gL = jnp.where(small, 1.0, gL)
    exact = jnp.expm1(safe_gL) / safe_gL
    taylor = 1.0 + gL / 2.0 + gL ** 2 / 6.0 + gL ** 3 / 24.0
    return jnp.where(small, taylor, exact)
