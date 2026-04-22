import jax.numpy as jnp
from jax import vmap, jit
from scipy.constants import c, m_e, m_p
from . import plasma
from .dispersion import _Zprime
from jax.scipy.special import gamma, gammaincc
from jax.scipy.signal import convolve
from .utility import reshape_moments
import matplotlib.pyplot as plt


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

    #Also normalize to the characteristic velocity vp
    xe = zetae * (jnp.sqrt(2 / 3 * gamma(5 / pe) / gamma(3 / pe)))
    xi = zetai * (jnp.sqrt(2 / 3 * gamma(5 / pi) / gamma(3 / pi)))

    # Calculate the susceptibilities
    chiE = jnp.zeros([efract.size, w.size], dtype=jnp.complex64)
    wpe = plasma.plasma_frequency(ne, 1, m_e / m_p)
    chiE = 2 * wpe**2 / (vTe * k)**2 * _Zprime(zetae, jnp.repeat(pe, jnp.shape(zetae)[-1], axis = -1))

    chiI = jnp.zeros([ifract.size, w.size], dtype=jnp.complex64)
    wpi = plasma.plasma_frequency(ni, ion_z, ion_a)
    chiI = 2 * wpi ** 2 / (vTi * k) ** 2 * _Zprime(zetai, jnp.repeat(pi, jnp.shape(zetai)[-1], axis = -1))

    #longitudinal dielectric function
    epsilon = 1 + jnp.sum(chiE, axis = 0) + jnp.sum(chiI, axis = 0)

    #electron and ion contributions to Skw
    econtr = efract * (
            2
            * jnp.pi
            / k
            / vTe
            / (2 * gamma(3 / pe))
            * (jnp.sqrt(2 / 3 * gamma(5 / pe) / gamma(3 / pe)))
            * jnp.power(jnp.abs(1 - jnp.sum(chiE, axis=0) / epsilon), 2)
            * gammaincc(2 / pe, jnp.abs(xe) ** pe)
            * gamma(2 / pe)
    )

    icontr = ifract * (
        2
        * jnp.pi
        * ion_z
        / k
        / vTi
        / (2 * gamma(3 / pi))
        * (jnp.sqrt(2 / 3 * gamma(5 / pi) / gamma(3 / pi)))
        * jnp.power(jnp.abs(jnp.sum(chiE, axis=0) / epsilon), 2)
        * gammaincc(2 / pi, jnp.abs(xi) ** pi)
        * gamma(2 / pi)
    )

    Skw = jnp.real(jnp.sum(econtr, axis = 0)+jnp.sum(icontr, axis = 0))

    return Skw.T

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
    Skw = _spectral_density(
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
        normalization_type = "max",
        normalization_scale = 1,
        notch = None,
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
    Skw = _spectral_density(
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
        scatter_vec,
        ue_dir,
        ui_dir
    )

    #Convert to wavelength space
    #Correction by dw/d(lambda) ~ lambda**(-2)
    Sklam = Skw / wavelengths[:, jnp.newaxis]**2

    #Now correct by (1+2w/wl) as given in Sheffield Eq. 5.1
    ws = 2 * jnp.pi * c / wavelengths
    wl = 2 * jnp.pi * c / probe_wavelength
    w = ws - wl

    w = w[:, jnp.newaxis]

    Pklam = Sklam * (1 + 2 * w / wl)

    # Here I assume that the instrument function is applied to the scattered power
    # and not to Skw, which I think is what the file I get from Joe Katz does...
    if instr_func_arr is not None:
        # Assuming a time-dependent instrument function, we use jax.vmap to apply the
        # relevant convolution to each time step
        # Not using 2D convolution to avoid time smearing
        Pklam = vmap(jnp.convolve, in_axes=1, out_axes=1)(Pklam, instr_func_arr, mode = "same")

    # Apply notch: NaN out wavelengths between notch[0] and notch[1]
    if notch is not None:
        mask = (wavelengths >= notch[0]) & (wavelengths <= notch[1])
        Pklam = jnp.where(mask[:, jnp.newaxis], jnp.nan, Pklam)

    #3 normalization options, based on setting the max, sum, or integral (dlambda) to 1
    # nan-safe: for integral, replace NaNs with 0 before integrating (notch contributes nothing)
    Pklam_finite = jnp.where(jnp.isnan(Pklam), 0.0, Pklam)
    normalization = normalization_scale * ((normalization_type=="max") / jnp.nanmax(Pklam, axis = 0)
                                           + (normalization_type=="sum") / jnp.nansum(Pklam, axis = 0)
                                           + (normalization_type=="integral") / jnp.trapezoid(Pklam_finite, wavelengths, axis = 0))

    Pklam *= normalization



    return Pklam


def scattered_power_wavelength(*args, **kwargs):

    return _scattered_power_wavelength(*args, **kwargs)
