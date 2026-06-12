"""Generate data_epw_2temp.h5 — an EPW streak with bi-Maxwellian electrons.

Run from the repo root:

    python ThomsonScatteringArbitrary/examples/data/make_data_epw_2temp.py

The electron species uses the user-supplied ``two_temp`` callable from
``examples/epw_custom_dist/my_dists.py`` (cold bulk + hot fraction), driven
through the general quadrature path. Pairs with
``examples/epw_custom_dist/fit.toml``.

Outputs (next to this script): ``Pkl_data``, ``Pkl_var``, ``wavelengths``,
``time``, plus ground truth ``ne_true_cm3``, ``Te_true_eV``, ``fhot_true``,
``rhot_true``.
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import e, k as kB

from ThomsonScatteringArbitrary.distributions import resolve_distribution
from ThomsonScatteringArbitrary.forward import scattered_power_wavelength

HERE = Path(__file__).resolve().parent
OUT = HERE / "data_epw_2temp.h5"

# ── Geometry: 263.25 nm probe, 60° scattering, antiStokes EPW window ─────────
probe_wavelength_m = 263.25e-9
probe_vec   = np.array([0.0, 0.0, 1.0])
scatter_vec = np.array([np.sin(np.deg2rad(60.0)), 0.0, np.cos(np.deg2rad(60.0))])
ue_dir = np.array([1.0, 0.0, 0.0])
ui_dir = np.array([1.0, 0.0, 0.0])

wavelengths_m = np.linspace(235e-9, 262e-9, 200)

# ── Plasma profiles ──────────────────────────────────────────────────────────
Nt = 8
time = np.linspace(0.0, 0.7, Nt)

ne_cm3 = np.full(Nt, 5e19)
Te_eV  = np.linspace(300.0, 450.0, Nt)
fhot   = np.linspace(0.05, 0.20, Nt)            # growing hot fraction
rhot   = np.full(Nt, 4.0)                       # hot/cold temperature ratio

ion_z = np.array([1])
ion_a = np.array([1])

two_temp = resolve_distribution(
    {"model": str(HERE.parent / "epw_custom_dist" / "my_dists.py") + ":two_temp",
     "x_max": 14.0, "n_points": 3001})
maxwellian = resolve_distribution("maxwellian")

Pkl_clean = np.asarray(scattered_power_wavelength(
    ne_cm3 * 1e6,
    np.zeros((1, Nt)), np.zeros((1, Nt)),
    (Te_eV * e / kB)[None, :], (100.0 * np.ones((1, Nt)) * e / kB),
    np.ones((1, Nt)), np.ones((1, Nt)),
    ion_z, ion_a,
    wavelengths_m, probe_wavelength_m,
    probe_vec, scatter_vec, ue_dir, ui_dir,
    e_models=(two_temp,), i_models=(maxwellian,),
    e_shapes=((fhot, rhot),), i_shapes=((),),
    normalization_type="max",
))

rng = np.random.default_rng(13)
noise_sigma = 0.02 * float(Pkl_clean.max())
Pkl_data = Pkl_clean + rng.normal(0.0, noise_sigma, Pkl_clean.shape)
Pkl_var  = np.full_like(Pkl_clean, noise_sigma ** 2)

with h5py.File(OUT, "w") as fh:
    fh.create_dataset("Pkl_data",    data=Pkl_data)
    fh.create_dataset("Pkl_var",     data=Pkl_var)
    fh.create_dataset("wavelengths", data=wavelengths_m)
    fh.create_dataset("time",        data=time)
    fh.create_dataset("ne_true_cm3", data=ne_cm3)
    fh.create_dataset("Te_true_eV",  data=Te_eV)
    fh.create_dataset("fhot_true",   data=fhot)
    fh.create_dataset("rhot_true",   data=rhot)

print(f"Wrote {OUT}  Pkl_data.shape={Pkl_data.shape}  "
      f"fhot: {fhot[0]:.2f}–{fhot[-1]:.2f}  Te: {Te_eV[0]:.0f}–{Te_eV[-1]:.0f} eV")
