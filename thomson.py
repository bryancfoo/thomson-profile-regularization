"""thomson.py — self-contained Thomson-scattering forward model for
*discretized* velocity distribution functions.

This file is intentionally standalone: it imports only third-party packages
(numpy, scipy) and **nothing** from the ``ThomsonScattering`` project.  Drop it
anywhere and it will run.

What it does
------------
Given a 1-D electron (and one or more ion) velocity distribution function(s)
supplied as **discrete numpy arrays** ``f(v)`` in SI units — *not* a normalized
``g(v*)`` on a dimensionless velocity — it computes the collective Thomson
scattering spectrum ``S(k, omega)`` and the scattered power in wavelength space
``P(k, lambda)``.

The distribution you pass is the 1-D distribution *reduced along the scattering
wavevector* ``k_hat`` (integrate the full 3-D f over the two velocity
components perpendicular to k).  ``v`` is that parallel velocity component in
m/s, in the lab frame, so any drift/flow along ``k_hat`` is simply baked into
where ``f`` sits on the ``v`` axis.  You do **not** pass a temperature or a
drift separately — the shape of ``f(v)`` is the whole story; only the density
(which sets the plasma frequency) and the ion charge/mass are needed as scalars.

Physics
-------
With ``v_phi = omega / k`` the physical phase velocity along ``k_hat``, the
susceptibility of species *s* is the Landau form

    chi_s(k, omega) = (wp_s^2 / k^2) * [ PV integral{ f_s'(v) / (v_phi - v) dv }
                                         + i*pi * f_s'(v_phi) ]

where ``f_s`` is normalized to unit area (``integral f dv = 1``) and
``wp_s^2 = n_s Z_s^2 e^2 / (m_s eps0)`` carries the absolute magnitude.  The
spectral density is (Sheffield, *Plasma Scattering of EM Radiation*, Ch. 5)

    S(k,omega) = sum_e (n_e/n) (2pi/k) |1 - chi_e/eps|^2 f_e(v_phi)
               + sum_i (Z_i^2 n_i/n) (2pi/k) |chi_e/eps|^2 f_i(v_phi),

    eps = 1 + chi_e + chi_i,   chi_e = sum over electron species,
                               chi_i = sum over ion species.

These formulas are algebraically identical to the ``ThomsonScattering`` package
(they are the same expressions after the substitution ``x=(v-u)/vth``,
``g(x)=vth*f(v)``); feeding a discretized Maxwellian here reproduces the
package's analytic Maxwellian to quadrature accuracy.

The pole (principal-value) integral is evaluated by singularity subtraction on
a uniform grid — see :func:`dispersion_integral` — so it is accurate even
though the input is a discrete array and even when ``v_phi`` lands between grid
points or far outside the thermal bulk.

Units summary (all SI)
----------------------
    velocity v           m/s
    distribution f(v)    s/m   (this file normalizes each f to unit area
                                internally; absolute scale comes from n)
    density n            m^-3
    wavelength           m     (e.g. 351 nm -> 351e-9)
    scattering angle     radians
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import pi
from scipy.constants import c, m_e, m_p, e, epsilon_0
from scipy.interpolate import CubicSpline
from scipy.special import gammaln

# numpy>=2.0 renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

__all__ = [
    "Species",
    "dispersion_integral",
    "spectral_density",
    "scattered_power",
    "thomson_spectrum",
    "scattering_angle_from_vectors",
    "maxwellian_f",
    "kappa_f",
    "bi_maxwellian_f",
    "super_gaussian_f",
]


# ─────────────────────────────────────────────────────────────────────────────
# Species container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Species:
    """One plasma population carrying a discretized 1-D distribution.

    Parameters
    ----------
    f : ndarray
        Reduced 1-D distribution along ``k_hat``.  Shape ``(Nt, Nv)`` or its
        transpose ``(Nv, Nt)`` (auto-detected against ``len(v)``), or ``(Nv,)``
        for a single, time-independent distribution.  Units are arbitrary and
        positive; the array is normalized to unit area internally.
    v : ndarray
        Velocity grid ``(Nv,)`` in m/s (component along ``k_hat``, lab frame).
        Need not be uniform; it is resampled internally.
    n : float or ndarray
        Number density of this population in m^-3.  Scalar or ``(Nt,)``.
    Z : float
        Charge number.  Use ``1`` for electrons (the sign is irrelevant — only
        ``Z^2`` enters), the ionization state for ions.
    mass : float
        Particle mass in kg.  Defaults to the electron mass; use
        :meth:`ion` to build an ion from mass number A.
    kind : str
        Descriptive tag ``"electron"`` / ``"ion"`` (set by the :meth:`electron`
        / :meth:`ion` factories).  Which dielectric screening factor a species
        gets is decided by whether it is passed in the ``electrons`` or the
        ``ions`` list, not by this tag.
    """

    f: np.ndarray
    v: np.ndarray
    n: object = 1.0
    Z: float = 1.0
    mass: float = m_e
    kind: str = "electron"

    @classmethod
    def electron(cls, f, v, n):
        """Electron population: Z=1, mass=m_e."""
        return cls(f=np.asarray(f, float), v=np.asarray(v, float),
                   n=n, Z=1.0, mass=m_e, kind="electron")

    @classmethod
    def ion(cls, f, v, n, Z=1.0, A=1.0):
        """Ion population with charge ``Z`` and mass number ``A`` (in amu)."""
        return cls(f=np.asarray(f, float), v=np.asarray(v, float),
                   n=n, Z=float(Z), mass=float(A) * m_p, kind="ion")


# ─────────────────────────────────────────────────────────────────────────────
# The pole integrator: chi integral with the principal-value singularity
# ─────────────────────────────────────────────────────────────────────────────
def _simpson_weights(n, dx):
    """Composite-Simpson weights for ``n`` (odd) uniform nodes of spacing dx."""
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (dx / 3.0)


class _PreparedDist:
    """A single-time distribution prepared for pole integration.

    Builds a cubic spline of the *area-normalized* ``f`` and a uniform Simpson
    grid of ``f'`` so that repeated queries at arbitrary phase velocities are
    cheap.  Outside ``[v.min, v.max]`` the distribution and its derivatives are
    treated as zero (compact support), which is what lets the dispersion
    integral reproduce the cold-plasma ``-wp^2/omega^2`` tail for phase
    velocities far out in the wings.
    """

    def __init__(self, v, f_row, n_grid=4001):
        v = np.asarray(v, float)
        f_row = np.asarray(f_row, float)

        # sort onto strictly increasing, unique velocity nodes
        order = np.argsort(v)
        v, f_row = v[order], f_row[order]
        v, uniq = np.unique(v, return_index=True)
        f_row = f_row[uniq]

        area = _trapz(f_row, v)
        self.valid = np.isfinite(area) and area > 0
        if not self.valid:
            # degenerate (all-zero / bad) distribution -> contributes nothing
            self.vmin, self.vmax = float(v[0]), float(v[-1])
            return
        f_row = f_row / area  # unit area: integral f dv = 1

        # C2 spline gives us f, f', f'' consistently; zero outside support.
        self._d0 = CubicSpline(v, f_row, extrapolate=False)
        self._d1 = self._d0.derivative(1)
        self._d2 = self._d0.derivative(2)

        self.vmin, self.vmax = float(v[0]), float(v[-1])

        # uniform quadrature grid for the Hilbert (principal-value) integral
        n_grid = int(n_grid)
        if n_grid % 2 == 0:
            n_grid += 1
        self.u = np.linspace(self.vmin, self.vmax, n_grid)
        self.du = self.u[1] - self.u[0]
        self.w_quad = _simpson_weights(n_grid, self.du)
        self.fp_grid = np.nan_to_num(self._d1(self.u), nan=0.0)

    # -- evaluation at arbitrary phase velocities (0 outside support) --------
    def value(self, vp):
        """Normalized distribution f(vp), zero outside the support."""
        if not self.valid:
            return np.zeros(np.shape(vp))
        return np.nan_to_num(self._d0(vp), nan=0.0)

    def Z(self, vp):
        """Dispersion integral Z(vp) = PV int f'(v)/(vp - v) dv + i*pi*f'(vp).

        This is the species-independent kernel of the susceptibility:
        ``chi = wp^2 / k^2 * Z(vp)``.
        """
        vp = np.asarray(vp, float)
        if not self.valid:
            return np.zeros(vp.shape, dtype=complex)

        shp = vp.shape
        q = vp.ravel()
        fp_q = np.nan_to_num(self._d1(q), nan=0.0)   # f'(vp)
        fpp_q = np.nan_to_num(self._d2(q), nan=0.0)   # f''(vp)

        # (Nq, Ng) reciprocal kernel 1/(vp - v), masked near the pole.
        denom = q[:, None] - self.u[None, :]
        eps = self.du / 2.0
        near = np.abs(denom) < eps
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(near, 0.0, 1.0 / np.where(near, 1.0, denom))

        # PV int (f'(v) - f'(vp)) / (vp - v) dv  by singularity subtraction:
        #   off-node nodes contribute the divided difference;
        #   nodes within du/2 of vp use the analytic limit -f''(vp).
        num = self.fp_grid[None, :] - fp_q[:, None]
        sum_off = np.sum(self.w_quad[None, :] * num * inv, axis=1)
        w_near = np.sum(near * self.w_quad[None, :], axis=1)
        integral = sum_off - fpp_q * w_near

        # analytic tail  f'(vp) * PV int_{vmin}^{vmax} 1/(vp - v) dv
        #              = f'(vp) * ln| (vp - vmin) / (vp - vmax) |
        tiny = 1e-300
        log_term = fp_q * (
            np.log(np.maximum(np.abs(q - self.vmin), tiny))
            - np.log(np.maximum(np.abs(q - self.vmax), tiny))
        )

        re = integral + log_term
        im = pi * fp_q
        return (re + 1j * im).reshape(shp)


def dispersion_integral(vp, v, f, n_grid=4001):
    """Chi integral with the principal-value pole, from a discretized f(v).

    Computes, for each query phase velocity ``vp``,

        Z(vp) = PV integral{ f'(v) / (vp - v) dv }  +  i*pi*f'(vp)

    where ``f`` (given on grid ``v``) is normalized to unit area first.  The
    susceptibility of a species is ``chi = wp^2 / k^2 * Z(vp)``.

    Parameters
    ----------
    vp : array_like
        Phase velocities ``omega/k`` (m/s) at which to evaluate.
    v : array_like, shape (Nv,)
        Velocity grid (m/s).  May be non-uniform.
    f : array_like, shape (Nv,)
        Discretized 1-D distribution (arbitrary positive units).
    n_grid : int
        Number of uniform Simpson nodes used for the principal-value integral.

    Returns
    -------
    ndarray (complex), same shape as ``vp``.
    """
    return _PreparedDist(v, f, n_grid=n_grid).Z(vp)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def scattering_angle_from_vectors(probe_vec, scatter_vec):
    """Scattering angle (radians) between probe and detection directions."""
    p = np.asarray(probe_vec, float)
    s = np.asarray(scatter_vec, float)
    p = p / np.linalg.norm(p)
    s = s / np.linalg.norm(s)
    return float(np.arccos(np.clip(np.dot(p, s), -1.0, 1.0)))


def _orient(f, Nv):
    """Coerce a species' f array to (Nt, Nv), auto-detecting the Nv axis."""
    f = np.asarray(f, float)
    if f.ndim == 1:
        if f.shape[0] != Nv:
            raise ValueError(f"1-D f has length {f.shape[0]}, expected {Nv}")
        return f[None, :]
    if f.ndim != 2:
        raise ValueError("f must be 1-D or 2-D")
    if f.shape[1] == Nv:
        return f
    if f.shape[0] == Nv:
        return f.T
    raise ValueError(f"neither axis of f {f.shape} matches len(v)={Nv}")


def _infer_Nt(species_list):
    """Common time length across species (arrays with Nt==1 broadcast)."""
    Nt = 1
    for sp in species_list:
        arr = _orient(sp.f, len(np.asarray(sp.v)))
        nt = arr.shape[0]
        n = np.ndim(sp.n)
        nt = max(nt, np.shape(sp.n)[0] if n == 1 else 1)
        if nt != 1:
            if Nt != 1 and Nt != nt:
                raise ValueError(f"inconsistent Nt: {Nt} vs {nt}")
            Nt = nt
    return Nt


# ─────────────────────────────────────────────────────────────────────────────
# Core forward model
# ─────────────────────────────────────────────────────────────────────────────
def spectral_density(
    wavelengths,
    probe_wavelength,
    scattering_angle,
    electrons,
    ions=(),
    n_grid=4001,
):
    """Collective Thomson spectral density ``S(k, omega)``.

    Parameters
    ----------
    wavelengths : ndarray (Nk,)
        Detection wavelengths in meters.
    probe_wavelength : float
        Probe (laser) wavelength in meters.
    scattering_angle : float
        Scattering angle in radians (use
        :func:`scattering_angle_from_vectors` to get it from geometry).
    electrons, ions : sequence of :class:`Species`
        Electron and ion populations.  Each carries its own ``f(v)`` array and
        velocity grid.  The total electron density ``n_e = sum(electron n)``
        sets the collective-mode weighting and the k-shift correction.
    n_grid : int
        Uniform nodes for the principal-value integral (see
        :func:`dispersion_integral`).

    Returns
    -------
    Skw : ndarray (Nk, Nt) real
        Spectral density on the wavelength grid, one column per time step.
    """
    electrons = list(electrons)
    ions = list(ions)
    if not electrons:
        raise ValueError("need at least one electron species")

    lam = np.asarray(wavelengths, float)
    Nk = lam.size
    Nt = _infer_Nt(electrons + ions)

    def _dens(n):
        return np.broadcast_to(np.asarray(n, float), (Nt,))

    ne_tot = np.zeros(Nt)
    for sp in electrons:
        ne_tot = ne_tot + _dens(sp.n)

    # ── geometry (per wavelength, per time via the density-dependent k) ──────
    ws = 2 * pi * c / lam            # scattered ang. frequency (Nk,)
    wl = 2 * pi * c / probe_wavelength
    w = ws - wl                      # frequency shift (Nk,)
    cos_th = np.cos(scattering_angle)

    # electron plasma frequency of the *total* electron density -> k shift
    wpe_tot_sq = ne_tot * e ** 2 / (m_e * epsilon_0)          # (Nt,)
    ks = np.sqrt(np.maximum(ws[None, :] ** 2 - wpe_tot_sq[:, None], 0.0)) / c
    kl = np.sqrt(np.maximum(wl ** 2 - wpe_tot_sq, 0.0)) / c   # (Nt,)
    k = np.sqrt(ks ** 2 + kl[:, None] ** 2
                - 2 * ks * kl[:, None] * cos_th)             # (Nt, Nk)
    vphi = w[None, :] / k                                     # (Nt, Nk)

    # ── per-time loop: distributions vary with time ──────────────────────────
    Skw = np.empty((Nk, Nt))
    for t in range(Nt):
        vp = vphi[t]      # (Nk,)
        k_t = k[t]        # (Nk,)

        # susceptibilities
        chi_e = np.zeros(Nk, dtype=complex)
        chi_i = np.zeros(Nk, dtype=complex)
        # cache (prepared distribution, wp^2, weight) per species for reuse
        e_prep, i_prep = [], []

        for sp in electrons:
            f_all = _orient(sp.f, len(np.asarray(sp.v)))
            f_row = f_all[min(t, f_all.shape[0] - 1)]
            prep = _PreparedDist(sp.v, f_row, n_grid=n_grid)
            wp2 = _dens(sp.n)[t] * sp.Z ** 2 * e ** 2 / (sp.mass * epsilon_0)
            weight = sp.Z ** 2 * _dens(sp.n)[t] / ne_tot[t]
            chi_e += wp2 / k_t ** 2 * prep.Z(vp)
            e_prep.append((prep, weight))

        for sp in ions:
            f_all = _orient(sp.f, len(np.asarray(sp.v)))
            f_row = f_all[min(t, f_all.shape[0] - 1)]
            prep = _PreparedDist(sp.v, f_row, n_grid=n_grid)
            wp2 = _dens(sp.n)[t] * sp.Z ** 2 * e ** 2 / (sp.mass * epsilon_0)
            weight = sp.Z ** 2 * _dens(sp.n)[t] / ne_tot[t]
            chi_i += wp2 / k_t ** 2 * prep.Z(vp)
            i_prep.append((prep, weight))

        eps = 1.0 + chi_e + chi_i
        e_screen = np.abs(1.0 - chi_e / eps) ** 2
        i_screen = np.abs(chi_e / eps) ** 2

        pref = 2 * pi / k_t
        col = np.zeros(Nk)
        for prep, weight in e_prep:
            col += weight * pref * e_screen * prep.value(vp)
        for prep, weight in i_prep:
            col += weight * pref * i_screen * prep.value(vp)
        Skw[:, t] = col

    return Skw


def scattered_power(
    wavelengths,
    probe_wavelength,
    scattering_angle,
    electrons,
    ions=(),
    n_grid=4001,
    irf=None,
    notch=None,
    normalization="none",
    normalization_scale=1.0,
):
    """Scattered power in wavelength space ``P(k, lambda)`` (what a streak sees).

    Same inputs as :func:`spectral_density`, plus the instrument pipeline:
    the frequency->wavelength Jacobian, an optional instrument response (IRF),
    an optional notch, and an optional per-column normalization.

    Parameters
    ----------
    irf : None, float, or ndarray
        Instrument response.  ``None`` = ideal.  A float is a Gaussian sigma in
        meters (same units as ``wavelengths``).  An array is a convolution
        kernel sampled on the wavelength pixel grid (it is area-normalized).
    notch : None or (lam_lo, lam_hi)
        Blank out wavelengths in ``[lam_lo, lam_hi]`` (meters) — e.g. a
        stray-light notch filter — before normalization.
    normalization : {"none", "max", "sum", "integral"}
        Per-time-column rescaling of the output.

    Returns
    -------
    Pklam : ndarray (Nk, Nt) real
    """
    lam = np.asarray(wavelengths, float)
    Skw = spectral_density(lam, probe_wavelength, scattering_angle,
                           electrons, ions, n_grid=n_grid)

    # frequency -> wavelength: dw/dlam ~ lam^-2, plus Sheffield Eq. 5.1 factor
    ws = 2 * pi * c / lam
    wl = 2 * pi * c / probe_wavelength
    w = ws - wl
    Pklam = Skw / lam[:, None] ** 2 * (1 + 2 * w[:, None] / wl)

    # instrument response (1-D convolution along wavelength, per time column)
    if irf is not None:
        if np.ndim(irf) == 0:
            kernel = _gaussian_kernel(float(irf), lam)
        else:
            kernel = np.asarray(irf, float)
            kernel = kernel / np.sum(kernel)
        Pklam = np.apply_along_axis(
            lambda coldata: np.convolve(coldata, kernel, mode="same"),
            axis=0, arr=Pklam)

    # notch: blank a wavelength band
    if notch is not None:
        mask = (lam >= notch[0]) & (lam <= notch[1])
        Pklam = np.where(mask[:, None], np.nan, Pklam)

    finite = np.where(np.isnan(Pklam), 0.0, Pklam)
    if normalization == "none":
        norm = normalization_scale
    elif normalization == "max":
        norm = normalization_scale / np.nanmax(finite, axis=0, keepdims=True)
    elif normalization == "sum":
        norm = normalization_scale / np.nansum(finite, axis=0, keepdims=True)
    elif normalization == "integral":
        norm = normalization_scale / _trapz(finite, lam, axis=0)[None, :]
    else:
        raise ValueError(f"unknown normalization {normalization!r}")

    return finite * norm


def _gaussian_kernel(sigma_m, lam):
    """Area-normalized Gaussian kernel on the wavelength pixel grid."""
    dlam = np.mean(np.diff(lam))
    half = max(1, int(np.ceil(4 * sigma_m / dlam)))
    x = np.arange(-half, half + 1) * dlam
    k = np.exp(-0.5 * (x / sigma_m) ** 2)
    return k / k.sum()


# ─────────────────────────────────────────────────────────────────────────────
# One-liner convenience wrapper: single electron + single ion population
# ─────────────────────────────────────────────────────────────────────────────
def thomson_spectrum(
    wavelengths,
    probe_wavelength,
    scattering_angle,
    fe, ve, ne,
    fi=None, vi=None,
    Zi=1.0, Ai=1.0,
    return_skw=False,
    **kwargs,
):
    """Convenience wrapper for the common one-electron / one-ion case.

    Parameters
    ----------
    fe, ve : ndarray
        Electron distribution ``(Nt, Nv)`` (or transpose / ``(Nv,)``) and its
        velocity grid ``(Nv,)`` in m/s.
    ne : float or ndarray
        Electron density (m^-3), scalar or ``(Nt,)``.
    fi, vi : ndarray, optional
        Ion distribution and velocity grid.  If omitted, only the (electron)
        feature is returned — useful for pure EPW work.  Ion density is set by
        quasineutrality: ``n_i = ne / Zi``.
    Zi, Ai : float
        Ion charge number and mass number (amu).
    return_skw : bool
        If True, return the raw ``S(k, omega)`` instead of the wavelength-space
        power.
    **kwargs :
        Forwarded to :func:`scattered_power` (``irf``, ``notch``,
        ``normalization``, ``normalization_scale``, ``n_grid``).

    Returns
    -------
    ndarray (Nk, Nt)
    """
    electrons = [Species.electron(fe, ve, ne)]
    ions = []
    if fi is not None:
        ne_arr = np.asarray(ne, float)
        ions = [Species.ion(fi, vi, ne_arr / Zi, Z=Zi, A=Ai)]

    if return_skw:
        n_grid = kwargs.get("n_grid", 4001)
        return spectral_density(wavelengths, probe_wavelength,
                                scattering_angle, electrons, ions,
                                n_grid=n_grid)
    return scattered_power(wavelengths, probe_wavelength, scattering_angle,
                           electrons, ions, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Distribution builders (handy for constructing discretized f(v) in SI units)
# ─────────────────────────────────────────────────────────────────────────────
def _vth(T_eV, mass):
    """Thermal speed vth = sqrt(2 T / m) with T in eV, m in kg."""
    return np.sqrt(2.0 * T_eV * e / mass)


def maxwellian_f(v, T_eV, mass, drift=0.0):
    """Maxwellian f(v) = exp(-((v-u)/vth)^2)/(sqrt(pi) vth),  integral f dv = 1.

    ``T_eV`` in eV, ``mass`` in kg, ``drift`` in m/s.
    """
    v = np.asarray(v, float)
    vth = _vth(T_eV, mass)
    x = (v - drift) / vth
    return np.exp(-x ** 2) / (np.sqrt(pi) * vth)


def super_gaussian_f(v, T_eV, mass, p=2.0, drift=0.0):
    """Projected isotropic super-Gaussian of order ``p`` (p=2 is Maxwellian).

    Matches the package's ``super_gaussian`` reduced distribution, expressed in
    physical velocity.  ``vth = sqrt(2 T/m)`` sets the scale.
    """
    from scipy.special import gamma, gammaincc
    v = np.asarray(v, float)
    vth = _vth(T_eV, mass)
    x = (v - drift) / vth
    g3, g5, g2 = gamma(3 / p), gamma(5 / p), gamma(2 / p)
    ratio = np.sqrt(2 / 3 * g5 / g3)
    xx = x * ratio
    g = ratio / (2 * g3) * g2 * gammaincc(2 / p, np.abs(xx) ** p)
    return g / vth


def kappa_f(v, T_eV, mass, kappa, drift=0.0):
    """1-D kappa distribution (Lorentzian-tailed), integral f dv = 1.

    Convention (matching the package): the kappa thermal speed equals
    ``vth = sqrt(2T/m)``; ``kappa`` must exceed 3/2.
    """
    v = np.asarray(v, float)
    vth = _vth(T_eV, mass)
    x = (v - drift) / vth
    norm = np.exp(gammaln(kappa) - gammaln(kappa - 0.5)) / np.sqrt(pi * kappa)
    g = norm * (1.0 + x ** 2 / kappa) ** (-kappa)
    return g / vth


def bi_maxwellian_f(v, T_eV, mass, fhot=0.1, rhot=4.0, drift=0.0):
    """Bi-Maxwellian: cold bulk at T plus a hot fraction ``fhot`` at ``rhot*T``.

    ``fhot`` is the hot *number* fraction; ``rhot`` the hot/cold temperature
    ratio.  integral f dv = 1.
    """
    cold = (1.0 - fhot) * maxwellian_f(v, T_eV, mass, drift)
    hot = fhot * maxwellian_f(v, rhot * T_eV, mass, drift)
    return cold + hot


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal smoke test: an IAW spectrum from discretized Maxwellians.
    theta = np.deg2rad(60.0)
    lam0 = 263.25e-9
    lam = np.linspace(262.8e-9, 263.7e-9, 400)

    ne = 6e19 * 1e6                      # 6e19 cm^-3 -> m^-3
    ve = np.linspace(-4e7, 4e7, 2000)    # electron velocities (m/s)
    vi = np.linspace(-1.2e6, 1.2e6, 2000)  # ion velocities (m/s)
    fe = maxwellian_f(ve, 500.0, m_e)
    fi = maxwellian_f(vi, 300.0, m_p)

    Pklam = thomson_spectrum(lam, lam0, theta, fe, ve, ne, fi, vi,
                             Zi=1.0, Ai=1.0, normalization="max")
    print("P(k,lam) shape:", Pklam.shape,
          " finite:", bool(np.all(np.isfinite(Pklam))),
          " peak at lam[nm] =", round(lam[np.argmax(Pklam[:, 0])] * 1e9, 3))
