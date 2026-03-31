"""
Test script for the regularized Thomson scattering fitter.

Generates a synthetic EPW streak from the profiles defined in forward_test.py,
adds 5% Gaussian noise, then scans over Tikhonov penalty weights and thresholds
using chi2_vary_tikhonov to produce an L-curve-style diagnostic plot.

Ion parameters (Ti, ui, pi, ifract) are held fixed at their true values.
Electron parameters (n, Te, ue) are fitted with 2nd-derivative Tikhonov
penalties. pe and efract are also free but unpenalized.
"""

import numpy as np
import matplotlib.pyplot as plt

from forward import _scattered_power_wavelength
from fitting import chi2_vary_tikhonov
from scipy.constants import e, k as kB


# ---------- True plasma profiles (forward_test.py lines 10-24) ----------
# Using actual time axis (not zeroed out) for meaningful time-varying profiles.

Nt = 51
t = np.linspace(0, 3, Nt)   # ns
tau = 1.5

ne_true    = 1e20 * np.exp(-t / tau)
ue_true    = 0e6  * np.exp(-np.sqrt(t / tau))
ui_true    = 0    * np.exp(-np.sqrt(t / tau))
Te_true    = 100  * np.exp(-t / tau)
Ti_true    = 200  - np.sqrt(t) * 100
ifractC    = 0.05 * t**2      # carbon charge fraction
ifractD    = 1 - ifractC      # deuterium charge fraction


# ---------- Measurement geometry (forward_test.py lines 30-45) ----------

def sph2cart(phi, theta):
    phi   = phi   / 180 * np.pi
    theta = theta / 180 * np.pi
    v = np.array([np.sin(phi) * np.cos(theta),
                  np.sin(phi) * np.sin(theta),
                  np.cos(phi)])
    return v / np.linalg.norm(v)

probe_wavelength = 263.25e-9   # m
epw_lam = np.linspace(263.25 - 30, 263.25 + 30, 1024) * 1e-9   # m

probe_vec   = sph2cart(116.57, 18)
scatter_vec = -sph2cart(112.15, 162)
k_vec       = scatter_vec - probe_vec
k_vec       = k_vec / np.linalg.norm(k_vec)


# ---------- Generate synthetic EPW spectrum ----------

Pkl_true = _scattered_power_wavelength(
    n               = ne_true * 1e6,
    Te              = np.array([Te_true]) * e / kB,
    Ti              = np.array([Ti_true, Ti_true]) * e / kB,
    ue              = np.array([ue_true]),
    ui              = np.array([ui_true, ui_true]),
    pe              = np.ones((1, Nt)) * 2,
    pi              = np.ones((2, Nt)) * 2,
    efract          = np.ones((1, Nt)),
    ifract          = np.array([ifractD, ifractC]),
    ion_z           = np.array([1.0, 6.0]),
    ion_a           = np.array([2.0, 12.0]),
    wavelengths     = epw_lam,
    probe_wavelength= probe_wavelength,
    probe_vec       = probe_vec,
    scatter_vec     = scatter_vec,
    ue_dir          = k_vec,
    ui_dir          = k_vec,
)
Pkl_true = np.array(Pkl_true)   # (Nk, Nt)


# ---------- Add 5% Gaussian noise ----------

rng        = np.random.default_rng(42)
noise_level = 0.05 * np.abs(Pkl_true)
Pkl_data   = Pkl_true + rng.normal(0, noise_level)
Pkl_var    = noise_level**2


# ---------- Measurement settings dict ----------

measurement_settings = {
    "Nelectrons"      : 1,
    "ion_z"           : np.array([1.0, 6.0]),
    "ion_a"           : np.array([2.0, 12.0]),
    "wavelengths"     : epw_lam,
    "probe_wavelength": probe_wavelength,
    "probe_vec"       : probe_vec,
    "scatter_vec"     : scatter_vec,
    "ue_dir"          : k_vec,
    "ui_dir"          : k_vec,
}


# ---------- params_settings ----------
# Electrons: flat initial guesses, free to vary.
# Ions: true time-varying profiles, held fixed (vary=False).

params_settings = {
    "n"      : {"value": 1e20,  "vary": True},
    "Te"     : {"value": 100.0, "vary": True},
    "ue"     : {"value": 0.0,   "vary": True},
    "pe"     : {"value": 2.0,   "vary": True},
    "efract" : {"value": 1.0,   "vary": True},
    # Ion species fixed at true profiles — one entry per (species, timestep).
    **{f"Ti0_{i}":     {"value": float(Ti_true[i]),  "vary": False} for i in range(Nt)},
    **{f"Ti1_{i}":     {"value": float(Ti_true[i]),  "vary": False} for i in range(Nt)},
    **{f"ui0_{i}":     {"value": float(ui_true[i]),  "vary": False} for i in range(Nt)},
    **{f"ui1_{i}":     {"value": float(ui_true[i]),  "vary": False} for i in range(Nt)},
    **{f"pi0_{i}":     {"value": 2.0,                "vary": False} for i in range(Nt)},
    **{f"pi1_{i}":     {"value": 2.0,                "vary": False} for i in range(Nt)},
    **{f"ifract0_{i}": {"value": float(ifractD[i]),  "vary": False} for i in range(Nt)},
    **{f"ifract1_{i}": {"value": float(ifractC[i]),  "vary": False} for i in range(Nt)},
}


# ---------- Penalty settings: 2nd-derivative Tikhonov on n, Te, ue ----------
# lambda_weights[k] weights the k-th derivative; indices 0 and 1 are zero
# so only the 2nd derivative is penalized.  Thresholds are absolute (relative=False).

base_penalty_settings = {
    "n" : {
        "profile_axis"  : t,
        "lambda_weights": [0.0, 0.0, 1.0],
        "thresholds"    : [0.0, 0.0, 1e17],   # ~0.1% of peak ne
        "relative"      : False,
    },
    "Te": {
        "profile_axis"  : t,
        "lambda_weights": [0.0, 0.0, 1.0],
        "thresholds"    : [0.0, 0.0, 1.0],    # 1 eV
        "relative"      : False,
    },
    "ue": {
        "profile_axis"  : t,
        "lambda_weights": [0.0, 0.0, 1.0],
        "thresholds"    : [0.0, 0.0, 1e4],    # 10 km/s
        "relative"      : False,
    },
}


# ---------- Scan penalty parameter space ----------

weight_scales = np.logspace(-2, 4, 7)   # 0.01 → 10000
cutoff_scales = np.logspace(-1, 1, 5)   # 0.1  → 10

chi2_grid = chi2_vary_tikhonov(
    Pkl_data, Pkl_var,
    measurement_settings,
    base_penalty_settings,
    weight_scales,
    cutoff_scales,
    params_settings=params_settings,
)


# ---------- Plot ----------

fig, ax = plt.subplots(figsize=(8, 5))
pcm = ax.pcolormesh(
    cutoff_scales, weight_scales, np.array(chi2_grid),
    shading="auto", cmap="viridis",
)
plt.colorbar(pcm, ax=ax, label="Chi2 (log likelihood, no regularization)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Threshold scale")
ax.set_ylabel("Weight scale")
ax.set_title("Chi2 vs Tikhonov penalty parameters")
plt.tight_layout()
plt.savefig("chi2_tikhonov_scan.png", dpi=150)
plt.show()
