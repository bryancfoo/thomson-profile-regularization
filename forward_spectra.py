"""forward_spectra.py — Thomson forward model with *per-time* distributions
**and** *per-time* velocity grids.

Same spirit and same physics as ``thomson.py`` (self-contained: numpy + scipy
only, nothing from the ``ThomsonScattering`` package), but generalized so that
**both** the distribution ``f`` **and** its velocity grid ``v`` may be 2-D
``(Nt, Nv)`` arrays that change from one time step to the next.  That lets you,
e.g., widen the ion velocity grid as the plasma heats, or feed distributions
sampled on a grid that drifts in time.

Everything you are likely to want to sweep is an explicit argument:

* the **measured** wavelength grid ``wavelengths`` and the **probe** (incident
  laser) wavelength ``probe_wavelength``;
* the scattering **k-vectors** ``probe_vec`` and ``scatter_vec`` (the geometry
  enters only through these);
* any number of ion species, each with its own charge ``Z`` and mass number
  ``A`` — the demo at the bottom uses two.

Array-shape convention (important)
----------------------------------
For every species, ``f`` and ``v`` may be

    1-D ``(Nv,)``       — a single grid/distribution shared across all times, or
    2-D ``(Nt, Nv)``    — one row per time step   (**axis 0 = time**,
                                                    **axis 1 = velocity**).

There is no transpose auto-detection here (unlike ``thomson.py``): when an array
is 2-D, axis 0 is time and the last axis is velocity, always.  A species' ``f``
and ``v`` must agree on ``Nv``; if ``v`` is 2-D it must match ``f`` row-for-row.

Physics (identical to ``thomson.py``)
-------------------------------------
With ``v_phi = omega/k`` the phase velocity along ``k_hat`` and ``f`` reduced
along ``k_hat`` and normalized to unit area,

    chi_s = (wp_s^2/k^2) [ PV int f_s'(v)/(v_phi - v) dv + i*pi*f_s'(v_phi) ],
    S(k,omega) = sum_e (n_e/n)(2pi/k)|1-chi_e/eps|^2 f_e(v_phi)
               + sum_i (Z_i^2 n_i/n)(2pi/k)|chi_e/eps|^2 f_i(v_phi),
    eps = 1 + chi_e + chi_i.

All SI: v [m/s], f [s/m] (normalized internally), n [m^-3], wavelength [m],
vectors dimensionless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy import pi
from scipy.constants import c, m_e, m_p, e, epsilon_0
from scipy.interpolate import CubicSpline
from scipy.special import gammaln

_trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed trapz

__all__ = [
    "Species",
    "forward_spectra",
    "dispersion_integral",
    "scattering_angle_from_vectors",
    "densities_from_fractions",
    "densities_from_charge_fractions",
    "maxwellian_f",
    "kappa_f",
]


# ─────────────────────────────────────────────────────────────────────────────
# Species container — f and v may each be (Nv,) or (Nt, Nv)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Species:
    """A plasma population with a (possibly time-varying) f and velocity grid.

    Parameters
    ----------
    f : ndarray
        1-D ``(Nv,)`` (shared across time) or 2-D ``(Nt, Nv)`` (axis 0 = time).
        Arbitrary positive units; normalized to unit area internally.
    v : ndarray
        Velocity grid along ``k_hat`` in m/s.  1-D ``(Nv,)`` (shared) or 2-D
        ``(Nt, Nv)`` (per-time grid, axis 0 = time).  Need not be uniform.
    n : float or ndarray
        Number density [m^-3]; scalar or ``(Nt,)``.
    Z : float
        Charge number (1 for electrons; ionization state for ions).
    mass : float
        Particle mass [kg].
    kind : str
        ``"electron"`` or ``"ion"`` (set by the factories); the screening term a
        species receives is decided by which list it is passed in, not by this.
    """

    f: object
    v: object
    n: object = 1.0
    Z: float = 1.0
    mass: float = m_e
    kind: str = "electron"

    @classmethod
    def electron(cls, f, v, n):
        return cls(f=np.asarray(f, float), v=np.asarray(v, float),
                   n=n, Z=1.0, mass=m_e, kind="electron")

    @classmethod
    def ion(cls, f, v, n, Z=1.0, A=1.0):
        """Ion with charge number ``Z`` and mass number ``A`` (amu)."""
        return cls(f=np.asarray(f, float), v=np.asarray(v, float),
                   n=n, Z=float(Z), mass=float(A) * m_p, kind="ion")

    # -- per-time-step access ------------------------------------------------
    def row(self, t):
        """(v_row, f_row) 1-D arrays for time index ``t`` (broadcast if 1-D)."""
        f = np.asarray(self.f, float)
        v = np.asarray(self.v, float)
        if v.ndim == 2:                       # per-time velocity grid
            if f.ndim != 2 or f.shape[-1] != v.shape[-1]:
                raise ValueError(
                    "a 2-D velocity grid needs f of matching (Nt, Nv)")
            return v[min(t, v.shape[0] - 1)], f[min(t, f.shape[0] - 1)]
        # v is 1-D (Nv,)
        if f.ndim == 1:
            return v, f
        if f.shape[-1] != v.shape[0]:
            raise ValueError(
                f"f last axis {f.shape[-1]} != len(v) {v.shape[0]}")
        return v, f[min(t, f.shape[0] - 1)]

    def nt(self):
        """Time length implied by this species (1 if fully time-independent)."""
        f = np.asarray(self.f)
        v = np.asarray(self.v)
        nt = 1
        if v.ndim == 2:
            nt = max(nt, v.shape[0])
        if f.ndim == 2:
            nt = max(nt, f.shape[0])
        if np.ndim(self.n) == 1:
            nt = max(nt, np.shape(self.n)[0])
        return nt


# ─────────────────────────────────────────────────────────────────────────────
# The pole integrator (principal-value chi integral), per time step
# ─────────────────────────────────────────────────────────────────────────────
def _simpson_weights(n, dx):
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * (dx / 3.0)


class _Prepared:
    """One (v_row, f_row) prepared for pole integration + feature evaluation."""

    def __init__(self, v, f_row, n_grid=4001):
        v = np.asarray(v, float)
        f_row = np.asarray(f_row, float)
        order = np.argsort(v)
        v, f_row = v[order], f_row[order]
        v, uniq = np.unique(v, return_index=True)
        f_row = f_row[uniq]

        area = _trapz(f_row, v)
        self.valid = bool(np.isfinite(area) and area > 0)
        self.vmin, self.vmax = float(v[0]), float(v[-1])
        if not self.valid:
            return
        f_row = f_row / area

        self._d0 = CubicSpline(v, f_row, extrapolate=False)
        self._d1 = self._d0.derivative(1)
        self._d2 = self._d0.derivative(2)

        n_grid = int(n_grid)
        if n_grid % 2 == 0:
            n_grid += 1
        self.u = np.linspace(self.vmin, self.vmax, n_grid)
        self.du = self.u[1] - self.u[0]
        self.w_quad = _simpson_weights(n_grid, self.du)
        self.fp_grid = np.nan_to_num(self._d1(self.u), nan=0.0)

    def value(self, vp):
        """Normalized f(vp), zero outside the support."""
        if not self.valid:
            return np.zeros(np.shape(vp))
        return np.nan_to_num(self._d0(vp), nan=0.0)

    def Z(self, vp):
        """PV int f'(v)/(vp-v) dv + i*pi*f'(vp)."""
        vp = np.asarray(vp, float)
        if not self.valid:
            return np.zeros(vp.shape, dtype=complex)
        shp = vp.shape
        q = vp.ravel()
        fp_q = np.nan_to_num(self._d1(q), nan=0.0)
        fpp_q = np.nan_to_num(self._d2(q), nan=0.0)

        denom = q[:, None] - self.u[None, :]
        eps = self.du / 2.0
        near = np.abs(denom) < eps
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(near, 0.0, 1.0 / np.where(near, 1.0, denom))
        num = self.fp_grid[None, :] - fp_q[:, None]
        sum_off = np.sum(self.w_quad[None, :] * num * inv, axis=1)
        w_near = np.sum(near * self.w_quad[None, :], axis=1)
        integral = sum_off - fpp_q * w_near

        tiny = 1e-300
        log_term = fp_q * (
            np.log(np.maximum(np.abs(q - self.vmin), tiny))
            - np.log(np.maximum(np.abs(q - self.vmax), tiny)))
        return (integral + log_term + 1j * pi * fp_q).reshape(shp)


def dispersion_integral(vp, v, f, n_grid=4001):
    """Chi integral with the principal-value pole, from a discretized f(v).

    ``Z(vp) = PV int f'(v)/(vp-v) dv + i*pi*f'(vp)`` (chi = wp^2/k^2 * Z).
    ``v``/``f`` are 1-D for a single time step.
    """
    return _Prepared(v, f, n_grid=n_grid).Z(vp)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry
# ─────────────────────────────────────────────────────────────────────────────
def scattering_angle_from_vectors(probe_vec, scatter_vec):
    """Scattering angle (radians) between probe and detection directions."""
    p = np.asarray(probe_vec, float)
    s = np.asarray(scatter_vec, float)
    p = p / np.linalg.norm(p)
    s = s / np.linalg.norm(s)
    return float(np.arccos(np.clip(np.dot(p, s), -1.0, 1.0)))


def densities_from_fractions(ne, number_fractions, ion_Z):
    """Per-ion densities [m^-3] from ion *number* fractions, quasineutral.

    Turns "what fraction of the ions is each species" into the absolute
    densities the model wants, enforcing charge neutrality
    ``sum_i Z_i n_i = n_e`` at every time step.  Any or all of the inputs may
    be time-dependent, so this is the natural way to give the model
    **time-varying ion fractions**.

    Parameters
    ----------
    ne : float or ndarray (Nt,)
        Electron density.
    number_fractions : sequence of (scalar or (Nt,))
        Relative ion *number* abundances, one per ion species.  They need not
        sum to 1 — they are renormalized.  Pass ``(Nt,)`` arrays to ramp the
        mix in time.
    ion_Z : sequence of float
        Charge number of each ion species (same order as ``number_fractions``).

    Returns
    -------
    list of ndarray (Nt,)
        Number density of each ion species.  With
        ``n_i = X * x_i`` and ``X = n_e / sum_i Z_i x_i`` (``x_i`` the
        normalized number fractions), ``sum_i Z_i n_i = n_e`` holds identically.
    """
    ne = np.asarray(ne, float)
    Zs = [float(z) for z in ion_Z]
    fr = [np.asarray(x, float) for x in number_fractions]
    if len(fr) != len(Zs):
        raise ValueError("number_fractions and ion_Z must have equal length")
    Nt = 1
    for a in [ne, *fr]:
        if a.ndim == 1:
            Nt = max(Nt, a.shape[0])
    ne_b = np.broadcast_to(ne, (Nt,))
    xb = [np.broadcast_to(x, (Nt,)) for x in fr]
    total = sum(xb)
    xn = [x / total for x in xb]                       # normalized fractions
    charge_per_ion = sum(z * x for z, x in zip(Zs, xn))  # sum_i Z_i x_i  (Nt,)
    X = ne_b / charge_per_ion                          # total ion number dens
    return [X * x for x in xn]


def densities_from_charge_fractions(ne, charge_fractions, ion_Z):
    """Per-ion densities [m^-3] from ion *charge* fractions ``Z_i n_i / n_e``.

    Charge fractions sum to 1 by quasineutrality; they are renormalized so
    ``sum_i Z_i n_i = n_e`` holds exactly, then ``n_i = q_i * n_e / Z_i``.
    Like :func:`densities_from_fractions`, any input may be ``(Nt,)`` to make
    the **ion fractions time-dependent**.

    Parameters
    ----------
    ne : float or ndarray (Nt,)
        Electron density.
    charge_fractions : sequence of (scalar or (Nt,))
        Fraction of the electron charge screened by each ion species,
        ``q_i = Z_i n_i / n_e``.  Need not sum to 1 — renormalized.
    ion_Z : sequence of float
        Charge number of each ion species (same order).

    Returns
    -------
    list of ndarray (Nt,)
        Number density of each ion species.
    """
    ne = np.asarray(ne, float)
    Zs = [float(z) for z in ion_Z]
    q = [np.asarray(x, float) for x in charge_fractions]
    if len(q) != len(Zs):
        raise ValueError("charge_fractions and ion_Z must have equal length")
    Nt = 1
    for a in [ne, *q]:
        if a.ndim == 1:
            Nt = max(Nt, a.shape[0])
    ne_b = np.broadcast_to(ne, (Nt,))
    qb = [np.broadcast_to(x, (Nt,)) for x in q]
    total = sum(qb)
    qn = [x / total for x in qb]                       # normalized -> sum to 1
    return [qi * ne_b / z for qi, z in zip(qn, Zs)]


# ─────────────────────────────────────────────────────────────────────────────
# Forward model
# ─────────────────────────────────────────────────────────────────────────────
def forward_spectra(
    wavelengths,
    probe_wavelength,
    probe_vec,
    scatter_vec,
    electrons,
    ions=(),
    n_grid=4001,
    irf=None,
    notch=None,
    normalization="none",
    normalization_scale=1.0,
    return_skw=False,
):
    """Time-resolved Thomson spectrum with per-time f and per-time velocity grids.

    Parameters
    ----------
    wavelengths : ndarray (Nk,)
        Measured (detected) wavelengths [m].
    probe_wavelength : float
        Incident/probe laser wavelength [m] (lambda_0).
    probe_vec, scatter_vec : (3,) array_like
        Propagation directions of the probe and scattered light — the geometry
        enters only through these (need not be unit vectors).
    electrons, ions : sequence of :class:`Species`
        Populations.  Each may carry 1-D or 2-D ``f`` and ``v`` (see the module
        docstring).  Total electron density ``n_e = sum(electron n)`` sets the
        collective weighting and the wavenumber shift.
    n_grid : int
        Uniform nodes for the principal-value integral.
    irf : None, float, or ndarray
        Instrument response: Gaussian sigma [m] (float) or convolution kernel
        (array), applied along wavelength.
    notch : None or (lam_lo, lam_hi)
        Blank a wavelength band [m] before normalization.
    normalization : {"none", "max", "sum", "integral"}
        Per-time-column rescaling.
    return_skw : bool
        Return raw ``S(k, omega)`` instead of wavelength-space power.

    Returns
    -------
    ndarray (Nk, Nt)
    """
    electrons = list(electrons)
    ions = list(ions)
    if not electrons:
        raise ValueError("need at least one electron species")

    lam = np.asarray(wavelengths, float)
    Nk = lam.size
    Nt = max(sp.nt() for sp in electrons + ions)

    def dens(n):
        return np.broadcast_to(np.asarray(n, float), (Nt,))

    ne_tot = np.zeros(Nt)
    for sp in electrons:
        ne_tot = ne_tot + dens(sp.n)

    # geometry (per wavelength; k also per time via the density shift)
    theta = scattering_angle_from_vectors(probe_vec, scatter_vec)
    cos_th = np.cos(theta)
    ws = 2 * pi * c / lam
    wl = 2 * pi * c / probe_wavelength
    w = ws - wl
    wpe_tot_sq = ne_tot * e ** 2 / (m_e * epsilon_0)          # (Nt,)
    ks = np.sqrt(np.maximum(ws[None, :] ** 2 - wpe_tot_sq[:, None], 0.0)) / c
    kl = np.sqrt(np.maximum(wl ** 2 - wpe_tot_sq, 0.0)) / c   # (Nt,)
    k = np.sqrt(ks ** 2 + kl[:, None] ** 2
                - 2 * ks * kl[:, None] * cos_th)             # (Nt, Nk)
    vphi = w[None, :] / k                                     # (Nt, Nk)

    Skw = np.empty((Nk, Nt))
    for t in range(Nt):
        vp, k_t = vphi[t], k[t]

        chi_e = np.zeros(Nk, dtype=complex)
        chi_i = np.zeros(Nk, dtype=complex)
        e_prep, i_prep = [], []

        for sp in electrons:
            v_row, f_row = sp.row(t)
            prep = _Prepared(v_row, f_row, n_grid=n_grid)
            wp2 = dens(sp.n)[t] * sp.Z ** 2 * e ** 2 / (sp.mass * epsilon_0)
            weight = sp.Z ** 2 * dens(sp.n)[t] / ne_tot[t]
            chi_e += wp2 / k_t ** 2 * prep.Z(vp)
            e_prep.append((prep, weight))

        for sp in ions:
            v_row, f_row = sp.row(t)
            prep = _Prepared(v_row, f_row, n_grid=n_grid)
            wp2 = dens(sp.n)[t] * sp.Z ** 2 * e ** 2 / (sp.mass * epsilon_0)
            weight = sp.Z ** 2 * dens(sp.n)[t] / ne_tot[t]
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

    if return_skw:
        return Skw

    # frequency -> wavelength (dw/dlam ~ lam^-2, plus Sheffield Eq. 5.1 factor)
    Pklam = Skw / lam[:, None] ** 2 * (1 + 2 * w[:, None] / wl)

    if irf is not None:
        if np.ndim(irf) == 0:
            kernel = _gaussian_kernel(float(irf), lam)
        else:
            kernel = np.asarray(irf, float)
            kernel = kernel / np.sum(kernel)
        Pklam = np.apply_along_axis(
            lambda col: np.convolve(col, kernel, mode="same"), 0, Pklam)

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
    dlam = np.mean(np.diff(lam))
    half = max(1, int(np.ceil(4 * sigma_m / dlam)))
    x = np.arange(-half, half + 1) * dlam
    kern = np.exp(-0.5 * (x / sigma_m) ** 2)
    return kern / kern.sum()


# ─────────────────────────────────────────────────────────────────────────────
# A couple of distribution builders (handy for constructing per-time arrays)
# ─────────────────────────────────────────────────────────────────────────────
def maxwellian_f(v, T_eV, mass, drift=0.0):
    """Maxwellian f(v), integral f dv = 1.  vth = sqrt(2T/m)."""
    v = np.asarray(v, float)
    vth = np.sqrt(2.0 * T_eV * e / mass)
    return np.exp(-((v - drift) / vth) ** 2) / (np.sqrt(pi) * vth)


def kappa_f(v, T_eV, mass, kappa, drift=0.0):
    """1-D kappa distribution (kappa > 3/2), integral f dv = 1."""
    v = np.asarray(v, float)
    vth = np.sqrt(2.0 * T_eV * e / mass)
    x = (v - drift) / vth
    norm = np.exp(gammaln(kappa) - gammaln(kappa - 0.5)) / np.sqrt(pi * kappa)
    return norm * (1.0 + x ** 2 / kappa) ** (-kappa) / vth


# ─────────────────────────────────────────────────────────────────────────────
# Demo: two ion species, time-varying distributions AND time-varying grids
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- inputs you can adjust ------------------------------------------------
    probe_wavelength = 263.25e-9                       # incident laser [m]
    wavelengths      = np.linspace(262.6e-9, 263.9e-9, 600)   # measured grid [m]
    # scattering k-vectors (60-degree scattering here)
    probe_vec   = [0.0, 0.0, 1.0]
    scatter_vec = [np.sin(np.deg2rad(60.0)), 0.0, np.cos(np.deg2rad(60.0))]

    ne = 6e19 * 1e6                                    # electron density [m^-3]
    Te = 600.0                                         # electron temperature [eV]

    # two ion species — adjust charge Z and mass number A freely
    ion1 = dict(Z=1.0, A=1.0)                          # hydrogen
    ion2 = dict(Z=6.0, A=12.0)                         # carbon C6+

    Nt, Nv = 14, 4001
    Ti1_t = np.linspace(120.0, 500.0, Nt)             # ion temperature ramps [eV]
    Ti2_t = np.linspace(120.0, 500.0, Nt)
    kap2_t = np.linspace(8.0, 2.0, Nt)                # carbon kappa index ramp

    # TIME-VARYING ion CHARGE fractions q_i = Z_i n_i / n_e (sum to 1 by
    # neutrality): carbon takes over 40% -> 90% of the electron charge.
    qC_t = np.linspace(0.40, 0.90, Nt)                # carbon charge fraction
    qH_t = 1.0 - qC_t                                 # hydrogen charge fraction
    n1_t, n2_t = densities_from_charge_fractions(
        ne, [qH_t, qC_t], [ion1["Z"], ion2["Z"]])     # each (Nt,) [m^-3]

    # electron background: fixed Maxwellian, shared across time (1-D f and v)
    ve = np.linspace(-1.2e8, 1.2e8, 8001)
    fe = maxwellian_f(ve, Te, m_e)

    # PER-TIME velocity grids: widen with sqrt(T) so the grid tracks the plasma.
    # vi1_t, vi2_t are 2-D (Nt, Nv); fi1_t, fi2_t are 2-D (Nt, Nv).
    def per_time_grid(Ti_t, mass, n_th=8.0):
        half = n_th * np.sqrt(2.0 * Ti_t * e / mass)          # (Nt,)
        return np.stack([np.linspace(-h, h, Nv) for h in half])

    vi1_t = per_time_grid(Ti1_t, ion1["A"] * m_p)
    vi2_t = per_time_grid(Ti2_t, ion2["A"] * m_p)
    fi1_t = np.stack([maxwellian_f(vi1_t[j], Ti1_t[j], ion1["A"] * m_p)
                      for j in range(Nt)])
    fi2_t = np.stack([kappa_f(vi2_t[j], Ti2_t[j], ion2["A"] * m_p, kap2_t[j])
                      for j in range(Nt)])

    print("electron f, v shapes:", fe.shape, ve.shape)
    print("ion 1  f, v shapes  :", fi1_t.shape, vi1_t.shape, "(per-time grid)")
    print("ion 2  f, v shapes  :", fi2_t.shape, vi2_t.shape, "(per-time grid)")

    electrons = [Species.electron(fe, ve, ne)]
    ions = [
        Species.ion(fi1_t, vi1_t, n1_t, Z=ion1["Z"], A=ion1["A"]),  # time-varying n
        Species.ion(fi2_t, vi2_t, n2_t, Z=ion2["Z"], A=ion2["A"]),  # time-varying n
    ]
    # number fraction n_i / (n1+n2), for reference against the charge fraction
    numfrac2 = n2_t / (n1_t + n2_t)
    print(f"carbon charge fraction : {qC_t[0]:.2f} -> {qC_t[-1]:.2f}")
    print(f"carbon number fraction : {numfrac2[0]:.2f} -> {numfrac2[-1]:.2f}")
    print(f"neutrality check  Z1 n1 + Z2 n2 - ne max = "
          f"{np.abs(ion1['Z']*n1_t + ion2['Z']*n2_t - ne).max():.2e}")

    P = forward_spectra(
        wavelengths, probe_wavelength, probe_vec, scatter_vec,
        electrons, ions, normalization="max",
    )
    print("spectrum P(k, lambda) shape (Nk, Nt):", P.shape,
          " finite:", bool(np.all(np.isfinite(P))))

    t = np.linspace(0.0, 1.3, Nt)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for j in [0, Nt // 2, Nt - 1]:
        ax[0].plot(vi1_t[j] / 1e6, fi1_t[j] * 1e6, "C0", alpha=0.4 + 0.5 * j / Nt)
        ax[0].plot(vi2_t[j] / 1e6, fi2_t[j] * 1e6, "C3", alpha=0.4 + 0.5 * j / Nt)
    ax[0].set_xlabel(r"$v$  [$10^6$ m/s]"); ax[0].set_ylabel(r"$f_i(v)$")
    ax[0].set_title("per-time ion grids widen with T (H blue, C red)")
    ax[0].grid(alpha=0.3)

    pm = ax[1].pcolormesh(t, (wavelengths - probe_wavelength) * 1e9, P,
                          shading="auto")
    fig.colorbar(pm, ax=ax[1], label="P (normalized per time)")
    ax[1].set_xlabel("time [ns]"); ax[1].set_ylabel(r"$\lambda-\lambda_0$  [nm]")
    ax[1].set_title("two-ion IAW streak (H + C)")
    plt.tight_layout()
    out = "forward_spectra_demo.png"
    plt.savefig(out, dpi=140)
    print("wrote", out)
