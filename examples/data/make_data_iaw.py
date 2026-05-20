"""Generate data_iaw.h5 — a 2-ion (D + C) IAW Thomson-scattering streak.

Run from anywhere (paths resolved relative to this script):

    python examples/data/make_data_iaw.py

Produces ``examples/data/data_iaw.h5`` with:
- ``Pkl_data`` (Nk, Nt)        : noisy scattered power
- ``Pkl_var``  (Nk, Nt)        : variance per pixel
- ``wavelengths`` (Nk,) meters : 262.6–263.9 nm window covering both IAW pairs
- ``time`` (Nt,)               : ns
- ``irf`` (Nk, Nt)             : per-time Gaussian IRF (FWHM ≈ 25 pm)
- ``ne_true_cm3, Te_true_eV, Ti_true_eV, ifrac_D_true, ifrac_C_true`` :
  ground-truth profiles for sanity-checking the recovered fit.

The data exercises features needed by ``examples/iaw_constraints/fit.toml``
(multi-ion sum-to-one constraint), ``examples/iaw_sample/fit.toml`` (same
plus posterior sampling), and ``examples/iaw_full/fit.toml`` (probe-beam
gain correction, background, IRF, notch, throughput).
"""
from pathlib import Path

import h5py
import numpy as np
from scipy.constants import e, k as kB

from ThomsonScattering.forward import scattered_power_wavelength

OUT = Path(__file__).resolve().parent / "data_iaw.h5"

# ── Geometry: 263.25 nm probe, 60° scattering ────────────────────────────────
probe_wavelength_nm = 263.25
probe_wavelength_m  = probe_wavelength_nm * 1e-9
probe_vec   = np.array([0.0, 0.0, 1.0])
scatter_vec = np.array([np.sin(np.deg2rad(60.0)), 0.0,
                        np.cos(np.deg2rad(60.0))])     # ≈ (0.866, 0, 0.5)
ue_dir = np.array([1.0, 0.0, 0.0])
ui_dir = np.array([1.0, 0.0, 0.0])

# ── Wavelength grid: tight IAW window centered on the probe line ────────────
# IAW peaks at Te=500 eV, Ti~200 eV land near ±0.24 nm (D) and ±0.10 nm (C).
# A 1.3-nm window with 200 points (~6.5 pm/pixel) resolves both pairs well.
wavelengths_nm = np.linspace(262.6, 263.9, 200)
wavelengths_m  = wavelengths_nm * 1e-9

# ── Plasma profiles (Nt time steps) — D + C plasma ──────────────────────────
Nt = 10
time = np.linspace(0.0, 0.9, Nt)                           # ns

ne_cm3   = np.linspace(5e19, 8e19, Nt)                     # 5–8e19 cm^-3
Te_eV    = np.full(Nt, 500.0)                              # constant 500 eV
Ti_D_eV  = np.linspace(200.0, 500.0, Nt)                   # ramping
Ti_C_eV  = np.linspace(150.0, 400.0, Nt)                   # ramping (cooler)

# Charge fractions: D=0.7, C=0.3 (sum to 1 — exercises the [constraints] table).
ifrac_D = np.full(Nt, 0.7)
ifrac_C = np.full(Nt, 0.3)

ue_ms = np.zeros(Nt)
ui_ms = np.zeros(Nt)
pe    = np.full(Nt, 2.0)
pi    = np.full(Nt, 2.0)
efract = np.ones(Nt)

ion_z = np.array([1, 6])                                   # D, C
ion_a = np.array([2, 12])

# Stack into (Nions, Nt) arrays
Ti_K     = np.vstack([Ti_D_eV, Ti_C_eV]) * e / kB
ui_stack = np.vstack([ui_ms, ui_ms])
pi_stack = np.vstack([pi, pi])
ifract   = np.vstack([ifrac_D, ifrac_C])

# Electrons: single species, in the shape (Nelectrons, Nt) the forward
# model expects.
Te_K    = (Te_eV * e / kB)[None, :]
ue_2d   = ue_ms[None, :]
pe_2d   = pe[None, :]
efract_2d = efract[None, :]

ne_m3 = ne_cm3 * 1e6

# ── Per-time Gaussian IRF (FWHM ≈ 25 pm) ─────────────────────────────────────
# Wavelength step is ≈6.5 pm; FWHM 25 pm ⇒ σ ≈ 10.6 pm ⇒ ~1.6 px.
sigma_pm = 25.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
dlam_pm = (wavelengths_nm[1] - wavelengths_nm[0]) * 1000.0
sigma_px = sigma_pm / dlam_pm
x = np.arange(len(wavelengths_nm)) - len(wavelengths_nm) // 2
irf_1d = np.exp(-0.5 * (x / sigma_px) ** 2)
irf_1d /= irf_1d.sum()
irf_arr = np.tile(irf_1d[:, None], (1, Nt))               # (Nk, Nt)

# ── Spectrometer throughput — gentle 5% envelope across the window ───────────
# A real instrument's transmission curve; baked into the data so the deck's
# `throughput = ...` line matches.  Also written as throughput.csv to
# demonstrate the CSV file-loading path.
throughput = 0.95 + 0.05 * np.exp(-((wavelengths_nm - 263.25) / 0.5) ** 2)
np.savetxt(Path(__file__).resolve().parent / "throughput.csv", throughput, fmt="%.6f")

Pkl_clean = np.asarray(scattered_power_wavelength(
    ne_m3,
    ue_2d, ui_stack,
    Te_K,  Ti_K,
    pe_2d, pi_stack,
    efract_2d, ifract,
    ion_z, ion_a,
    wavelengths_m, probe_wavelength_m,
    probe_vec, scatter_vec, ue_dir, ui_dir,
    instr_func_arr=irf_arr,           # bake spectrometer IRF into the data
    irf_normalization="area",
    throughput=throughput,            # bake spectrometer throughput in too
    normalization_type="max",
))   # (Nk, Nt)

rng = np.random.default_rng(7)
noise_sigma = 0.02 * float(Pkl_clean.max())
Pkl_data = Pkl_clean + rng.normal(0.0, noise_sigma, Pkl_clean.shape)
Pkl_var  = np.full_like(Pkl_clean, noise_sigma ** 2)

with h5py.File(OUT, "w") as fh:
    fh.create_dataset("Pkl_data",     data=Pkl_data)
    fh.create_dataset("Pkl_var",      data=Pkl_var)
    fh.create_dataset("wavelengths",  data=wavelengths_m)
    fh.create_dataset("time",         data=time)
    fh.create_dataset("irf",          data=irf_arr)
    fh.create_dataset("throughput",   data=throughput)
    fh.create_dataset("ne_true_cm3",  data=ne_cm3)
    fh.create_dataset("Te_true_eV",   data=Te_eV)
    fh.create_dataset("Ti_D_true_eV", data=Ti_D_eV)
    fh.create_dataset("Ti_C_true_eV", data=Ti_C_eV)
    fh.create_dataset("ifrac_D_true", data=ifrac_D)
    fh.create_dataset("ifrac_C_true", data=ifrac_C)

print(
    f"Wrote {OUT}  Pkl_data.shape={Pkl_data.shape}  "
    f"ne: {ne_cm3[0]:.1e}–{ne_cm3[-1]:.1e} cm^-3  "
    f"Ti_D: {Ti_D_eV[0]:.0f}–{Ti_D_eV[-1]:.0f} eV  "
    f"λ: {wavelengths_nm[0]:.1f}–{wavelengths_nm[-1]:.1f} nm"
)
