"""Generate example_data.h5 for use with example_deck.toml.

Run this once before running the example fit:
    python make_example_data.py

Then run the fit:
    python fit_from_deck.py
    > Path to input deck (.toml): example_deck.toml
"""

import numpy as np
import h5py
from scipy.constants import k as kB, e

from ThomsonScattering.forward import scattered_power_wavelength

# ── Geometry ─────────────────────────────────────────────────────────────────
probe_wavelength = 263.25               # nm
probe_vec   = np.array([0.0, 0.0, 1.0])
scatter_vec = np.array([1.0, 0.0, 0.0])
ue_dir      = np.array([1.0, 0.0, 0.0])
ui_dir      = np.array([1.0, 0.0, 0.0])

# ── Wavelength grid ───────────────────────────────────────────────────────────
wavelengths = np.linspace(261.0, 266.0, 256)   # nm

# ── True plasma profiles (Nt time steps) in fitting-param units ───────────────
# n  in cm^-3   (fitter stores cm^-3, forward model uses cm^-3 * 1e6 = m^-3)
# T  in eV      (fitter stores eV, forward model uses eV * e/kB = K)
Nt = 10
time = np.linspace(0.0, 1.0, Nt)

ne_cm3  = np.linspace(4e18, 8e18, Nt)          # cm^-3
Te_eV   = np.linspace(300.0, 700.0, Nt)        # eV
Ti_eV   = np.linspace(100.0, 250.0, Nt)        # eV (same for both ion species)
ue_ms   = np.zeros(Nt)                          # m/s
ui_ms   = np.zeros(Nt)                          # m/s

# Two ion species: deuterium (D, z=1, a=2) + carbon (C, z=6, a=12)
# ifract = ion charge fraction (not number fraction)
ion_z    = np.array([1, 6])
ion_a    = np.array([2, 12])
ifractD  = np.full(Nt, 0.8)
ifractC  = np.full(Nt, 0.2)

# Single electron species
efract   = np.ones(Nt)
pe       = np.full(Nt, 2.0)    # Maxwellian (super-Gaussian exponent = 2)
pi       = np.full(Nt, 2.0)

# ── Convert to units expected by the forward model ───────────────────────────
ne_m3  = ne_cm3 * 1e6                          # cm^-3 → m^-3
Te_K   = Te_eV  * e / kB                       # eV → K
Ti_K   = Ti_eV  * e / kB

# ── Stack into (Nspecies, Nt) arrays ─────────────────────────────────────────
Te_arr     = Te_K[np.newaxis, :]               # (1, Nt)
Ti_arr     = np.stack([Ti_K, Ti_K], axis=0)    # (2, Nt)
ue_arr     = ue_ms[np.newaxis, :]              # (1, Nt)
ui_arr     = np.stack([ui_ms, ui_ms], axis=0)  # (2, Nt)
pe_arr     = pe[np.newaxis, :]                 # (1, Nt)
pi_arr     = np.stack([pi, pi], axis=0)        # (2, Nt)
efract_arr = efract[np.newaxis, :]             # (1, Nt)
ifract_arr = np.stack([ifractD, ifractC], axis=0)  # (2, Nt)

# ── Compute clean synthetic spectrum ─────────────────────────────────────────
Pkl_clean = np.asarray(scattered_power_wavelength(
    ne_m3, ue_arr, ui_arr, Te_arr, Ti_arr, pe_arr, pi_arr,
    efract_arr, ifract_arr,
    ion_z, ion_a, wavelengths * 1e-9, probe_wavelength * 1e-9,
    probe_vec, scatter_vec, ue_dir, ui_dir,
    normalization_type="max",
))  # shape (Nk, Nt)

# ── Add Gaussian noise ────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
noise_sigma = 0.03 * float(Pkl_clean.max())
Pkl_data = Pkl_clean + rng.normal(0.0, noise_sigma, Pkl_clean.shape)
Pkl_var  = np.full_like(Pkl_clean, noise_sigma ** 2)

# ── Save to HDF5 ──────────────────────────────────────────────────────────────
out_path = "example_data.h5"
with h5py.File(out_path, "w") as fh:
    fh.create_dataset("Pkl_data",    data=Pkl_data)
    fh.create_dataset("Pkl_var",     data=Pkl_var)
    fh.create_dataset("wavelengths", data=wavelengths)
    fh.create_dataset("time",        data=time)
    # True profiles stored for post-fit comparison
    fh.create_dataset("ne_true_cm3", data=ne_cm3)
    fh.create_dataset("Te_true_eV",  data=Te_eV)
    fh.create_dataset("Ti_true_eV",  data=Ti_eV)

print(
    f"Wrote {out_path}  "
    f"Pkl_data.shape={Pkl_data.shape}  "
    f"ne: {ne_cm3[0]:.1e}–{ne_cm3[-1]:.1e} cm-3  "
    f"Te: {Te_eV[0]:.0f}–{Te_eV[-1]:.0f} eV"
)

# ── Plot clean vs noisy spectra ───────────────────────────────────────────────
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

vmin = min(Pkl_clean.min(), Pkl_data.min())
vmax = max(Pkl_clean.max(), Pkl_data.max())

for ax, data, title in zip(axes, [Pkl_clean, Pkl_data], ["Pkl_clean", "Pkl_data"]):
    mesh = ax.pcolormesh(time, wavelengths, data, vmin=vmin, vmax=vmax, shading="auto")
    ax.set_xlabel("Time")
    ax.set_title(title)

axes[0].set_ylabel("Wavelength (nm)")
fig.colorbar(mesh, ax=axes, label="Scattered power (norm.)")
#fig.savefig("synthetic_thomson.png", dpi=150, bbox_inches="tight")
plt.show()
#print("Saved synthetic_thomson.png")
