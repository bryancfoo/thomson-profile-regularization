"""Generate data_epw.h5 for the EPW examples.

Run from anywhere (paths are resolved relative to this script):

    python examples/data/make_data_epw.py

The output, ``examples/data/data_epw.h5``, contains a synthetic time-resolved
Thomson scattering streak in the antiStokes EPW window for a 263.25 nm
probe with 60° scattering.  Only ne and Te vary with time; everything
else is held fixed.  Used by ``examples/forward_only/forward.toml`` and
``examples/epw_basic/fit.toml``.
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import e, k as kB

from ThomsonScattering.forward import scattered_power_wavelength

OUT = Path(__file__).resolve().parent / "data_epw.h5"

# ── Geometry: 263.25 nm probe, 60° scattering ────────────────────────────────
# Codebase convention: wavelengths and probe_wavelength are in METERS.  We
# work in nm for readability and convert once before saving / before calling
# the forward model.
probe_wavelength_nm = 263.25
probe_wavelength_m  = probe_wavelength_nm * 1e-9
probe_vec   = np.array([0.0, 0.0, 1.0])
# 60° between probe and scatter directions.  cos(60°) = 0.5.
scatter_vec = np.array([np.sin(np.deg2rad(60.0)), 0.0,
                        np.cos(np.deg2rad(60.0))])     # ≈ (0.8660, 0, 0.5)
ue_dir      = np.array([1.0, 0.0, 0.0])
ui_dir      = np.array([1.0, 0.0, 0.0])

# ── Wavelength grid: antiStokes EPW window only ──────────────────────────────
# At 60° / 263.25 nm with Te ∈ [200, 400] eV and ne ∈ [3e19, 7e19] cm^-3, the
# antiStokes EPW peak roams between roughly 242 and 249 nm.  Take 235–262 nm
# so the line is well inside the window across the full streak.  Stored in
# METERS to match the codebase convention.
wavelengths_nm = np.linspace(235.0, 262.0, 200)
wavelengths_m  = wavelengths_nm * 1e-9                  # Nk=200

# ── Plasma profiles (Nt time steps); Te, ne vary; everything else fixed ──────
Nt = 10
time = np.linspace(0.0, 0.9, Nt)                        # ns
ne_cm3 = np.linspace(3e19, 7e19, Nt)                    # cm^-3
Te_eV  = np.linspace(200.0, 400.0, Nt)                  # eV

Ti_eV  = np.full(Nt, 100.0)                             # fixed; EPW barely sees it
ue_ms  = np.zeros(Nt)
ui_ms  = np.zeros(Nt)
pe     = np.full(Nt, 2.0)                               # Maxwellian
pi     = np.full(Nt, 2.0)
efract = np.ones(Nt)
ifract = np.ones(Nt)                                    # single ion species

# Single ion species: protons (z=1, a=1).  EPW shape depends only weakly on
# the ions, so a single species keeps the parameter count minimal.
ion_z = np.array([1])
ion_a = np.array([1])

# ── Convert to forward-model units (m^-3, K) ─────────────────────────────────
ne_m3 = ne_cm3 * 1e6
Te_K  = Te_eV  * e / kB
Ti_K  = Ti_eV  * e / kB

Pkl_clean = np.asarray(scattered_power_wavelength(
    ne_m3,
    ue_ms[None, :], ui_ms[None, :],
    Te_K[None, :],  Ti_K[None, :],
    pe[None, :],    pi[None, :],
    efract[None, :], ifract[None, :],
    ion_z, ion_a,
    wavelengths_m, probe_wavelength_m,
    probe_vec, scatter_vec, ue_dir, ui_dir,
    normalization_type="max",
))   # (Nk, Nt)

# ── Add Gaussian noise; record variance so the fitter can use it ─────────────
rng = np.random.default_rng(42)
noise_sigma = 0.03 * float(Pkl_clean.max())
Pkl_data = Pkl_clean + rng.normal(0.0, noise_sigma, Pkl_clean.shape)
Pkl_var  = np.full_like(Pkl_clean, noise_sigma ** 2)

with h5py.File(OUT, "w") as fh:
    fh.create_dataset("Pkl_data",    data=Pkl_data)
    fh.create_dataset("Pkl_var",     data=Pkl_var)
    # Wavelengths stored in METERS — matches the codebase convention used
    # by both forward.py (computes 2πc/wavelengths directly) and the
    # deck-loading path (no nm→m conversion happens on load).
    fh.create_dataset("wavelengths", data=wavelengths_m)
    fh.create_dataset("time",        data=time)
    fh.create_dataset("ne_true_cm3", data=ne_cm3)
    fh.create_dataset("Te_true_eV",  data=Te_eV)

print(
    f"Wrote {OUT}  Pkl_data.shape={Pkl_data.shape}  "
    f"ne: {ne_cm3[0]:.1e}–{ne_cm3[-1]:.1e} cm^-3  "
    f"Te: {Te_eV[0]:.0f}–{Te_eV[-1]:.0f} eV  "
    f"λ: {wavelengths_nm[0]:.0f}–{wavelengths_nm[-1]:.0f} nm (stored in m)"
)
