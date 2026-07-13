"""Generate data_kappa.h5 — an IAW streak with a kappa (suprathermal) ion species.

Run from the repo root:

    python examples/data/make_data_kappa.py

Produces ``data_kappa.h5`` next to this script with:
- ``Pkl_data`` (Nk, Nt)        : noisy scattered power
- ``Pkl_var``  (Nk, Nt)        : variance per pixel
- ``wavelengths`` (Nk,) meters : 262.6–263.9 nm IAW window
- ``time`` (Nt,)               : ns
- ``ne_true_cm3, Te_true_eV, Ti_true_eV, kappa_true`` : ground truth

Single H ion species with a kappa velocity distribution (kappa ramping
4 → 2.2 over the streak: increasingly suprathermal tails), Maxwellian
electrons. Pairs with ``examples/iaw_kappa/fit.toml``.
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import e, k as kB

from ThomsonScattering.distributions import resolve_distribution
from ThomsonScattering.forward import scattered_power_wavelength

OUT = Path(__file__).resolve().parent / "data_kappa.h5"

# ── Geometry: 263.25 nm probe, 60° scattering (matches the IAW examples) ─────
probe_wavelength_m = 263.25e-9
probe_vec   = np.array([0.0, 0.0, 1.0])
scatter_vec = np.array([np.sin(np.deg2rad(60.0)), 0.0, np.cos(np.deg2rad(60.0))])
ue_dir = np.array([1.0, 0.0, 0.0])
ui_dir = np.array([1.0, 0.0, 0.0])

wavelengths_m = np.linspace(262.6e-9, 263.9e-9, 200)

# ── Plasma profiles ──────────────────────────────────────────────────────────
Nt = 8
time = np.linspace(0.0, 0.7, Nt)                # ns

ne_cm3 = np.full(Nt, 6e19)
Te_eV  = np.full(Nt, 500.0)
Ti_eV  = np.linspace(200.0, 350.0, Nt)
kappa  = np.linspace(4.0, 2.2, Nt)              # increasingly suprathermal

ion_z = np.array([1])
ion_a = np.array([1])

maxwellian = resolve_distribution("maxwellian")
kappa_model = resolve_distribution("kappa")

Pkl_clean = np.asarray(scattered_power_wavelength(
    ne_cm3 * 1e6,
    np.zeros((1, Nt)), np.zeros((1, Nt)),                  # ue, ui
    (Te_eV * e / kB)[None, :], (Ti_eV * e / kB)[None, :],  # Te, Ti
    np.ones((1, Nt)), np.ones((1, Nt)),                    # efract, ifract
    ion_z, ion_a,
    wavelengths_m, probe_wavelength_m,
    probe_vec, scatter_vec, ue_dir, ui_dir,
    e_models=(maxwellian,), i_models=(kappa_model,),
    e_shapes=((),), i_shapes=((kappa,),),
    normalization_type="max",
))

rng = np.random.default_rng(11)
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
    fh.create_dataset("Ti_true_eV",  data=Ti_eV)
    fh.create_dataset("kappa_true",  data=kappa)

print(f"Wrote {OUT}  Pkl_data.shape={Pkl_data.shape}  "
      f"kappa: {kappa[0]:.1f}–{kappa[-1]:.1f}  Ti: {Ti_eV[0]:.0f}–{Ti_eV[-1]:.0f} eV")
