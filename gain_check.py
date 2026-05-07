"""gain_check.py — sanity check for the Turnbull SRS/SBS gain correction.

Synthetic three-population deuterium plasma at ne = 1e20 cm^-3:
  - bulk at rest                                     (80% charge fraction)
  - +1000 km/s tail, decaying linearly to 0 by 3 ns  (10%)
  - -1000 km/s tail, decaying linearly to 0 by 3 ns  (10%)

Te is held flat at 500 eV. Ti starts cold (50 eV), ramps to 600 eV at 2 ns,
then decays slowly to 450 eV at 5 ns.

Forward-models the same plasma twice — once with gain_mode="off" and once
with gain_mode="exact" — at probe wavelength 263.25 nm and 60 degree
scattering angle. Plots both streaks plus their difference so the
Stokes amplification / anti-Stokes depletion is easy to see.

Run:
    python gain_check.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import e, k as kB
import jax.numpy as jnp

from ThomsonScattering.forward import scattered_power_wavelength, _spectral_density
from ThomsonScattering import gain as _gain_mod
from ThomsonScattering.utility import reshape_moments


# ── Geometry ─────────────────────────────────────────────────────────────────
probe_wavelength_nm  = 263.25
scattering_angle_deg = 60.0

probe_vec = np.array([0.0, 0.0, 1.0])
theta     = np.deg2rad(scattering_angle_deg)
scatter_vec = np.array([np.sin(theta), 0.0, np.cos(theta)])

# Flow direction = k-vector direction so flow magnitude maps directly to
# Doppler shift on the spectrum.
k_hat   = scatter_vec - probe_vec
k_hat  /= np.linalg.norm(k_hat)
ue_dir  = k_hat
ui_dir  = k_hat


# ── Time and wavelength grids ────────────────────────────────────────────────
Nt = 80
t_ns = np.linspace(0.0, 5.0, Nt)

# Focus on the IAW band around the laser line (EPW peaks sit at +-20 nm here
# and aren't relevant to the flow asymmetry being demonstrated).
Nk = 256
wavelengths_nm = np.linspace(probe_wavelength_nm - 2.0,
                             probe_wavelength_nm + 2.0, Nk)


# ── Plasma profiles ──────────────────────────────────────────────────────────
ne_cm3 = np.full(Nt, 1.0e20)
Te_eV  = np.full(Nt, 500.0)

# ── Ion temperature shape parameters ─────────────────────────────────────────
t_peak_ns    =  3.0    # time of Ti peak (ns)

Ti0_cold_eV  =   50.0  # bulk — initial
Ti0_peak_eV  = 1000.0  # bulk — peak
Ti0_final_eV =  800.0  # bulk — end of window

Ti1_cold_eV  =   2500.0  # fast ions — initial
Ti1_peak_eV  = 3000.0  # fast ions — peak
Ti1_final_eV =  500.0  # fast ions — end of window

def _make_Ti(cold, peak, final, t_ns, t_peak):
    ramp  = cold * (peak / cold) ** (t_ns / t_peak)
    decay = peak - (peak - final) * (t_ns - t_peak) / (t_ns[-1] - t_peak)
    return np.where(t_ns < t_peak, ramp, decay)

Ti0_eV = _make_Ti(Ti0_cold_eV, Ti0_peak_eV, Ti0_final_eV, t_ns, t_peak_ns)
Ti1_eV = _make_Ti(Ti1_cold_eV, Ti1_peak_eV, Ti1_final_eV, t_ns, t_peak_ns)

# Three deuterons (Z=1, A=2). Same Z everywhere ⇒ charge fraction == number fraction.
ion_z = np.array([1, 1, 1])
ion_a = np.array([2, 2, 2])

ifract_arr = np.stack([
    np.full(Nt, 0.6),   # bulk
    np.full(Nt, 0.2),   # +flow
    np.full(Nt, 0.2),   # -flow
])

# Flow velocities (m/s): bulk = 0; +-1000 km/s tails ramp linearly to 0 over 3 ns.
v_max = 1.0e6
ramp  = np.clip(1.0 - t_ns / 3.0, 0.0, 1.0)
ui_arr = np.stack([
    np.zeros(Nt),
    +v_max * ramp,
    -v_max * ramp,
])

Ti_arr_eV  = np.stack([Ti0_eV, Ti1_eV, Ti1_eV])  # bulk, +flow, -flow
Te_arr_eV  = Te_eV[np.newaxis, :]                # (1, Nt)
ue_arr     = np.zeros((1, Nt))
pe_arr     = np.full((1, Nt), 2.0)
pi_arr     = np.full((3, Nt), 2.0)
efract_arr = np.ones((1, Nt))

# Convert to forward-model units: density m^-3, temperature K.
ne_m3 = ne_cm3 * 1e6
Te_K  = Te_arr_eV  * (e / kB)
Ti_K  = Ti_arr_eV  * (e / kB)


# ── Probe-beam parameters for the gain correction ────────────────────────────
probe_intensity = 5.0e15      # W/cm^2
probe_diameter  = 200e-6      # m  (200 um FWHM)
pol_p_fraction  = 1.0         # pure p-pol (in scattering plane)


# ── Run forward model with and without the gain correction ───────────────────
common = dict(
    n=ne_m3, ue=ue_arr, ui=ui_arr,
    Te=Te_K, Ti=Ti_K,
    pe=pe_arr, pi=pi_arr,
    efract=efract_arr, ifract=ifract_arr,
    ion_z=ion_z, ion_a=ion_a,
    wavelengths=wavelengths_nm * 1e-9,
    probe_wavelength=probe_wavelength_nm * 1e-9,
    probe_vec=probe_vec, scatter_vec=scatter_vec,
    ue_dir=ue_dir, ui_dir=ui_dir,
    normalization_type="max",
)

print("Running forward model (no gain) ...")
Pkl_no_gain = np.asarray(scattered_power_wavelength(**common, gain_mode="off"))

print("Running forward model (with gain) ...")
Pkl_gain = np.asarray(scattered_power_wavelength(
    **common,
    probe_intensity=probe_intensity,
    probe_diameter=probe_diameter,
    pol_p_fraction=pol_p_fraction,
    gain_mode="exact",
))


# ── Compute G(λ, t) directly from susceptibilities ───────────────────────────
print("Computing gain factor profile ...")
_Ni = ifract_arr.shape[0]
_Ne = efract_arr.shape[0]

sd_out = _spectral_density(
    reshape_moments(ne_m3,      _Ni, Nt),
    reshape_moments(ue_arr,     _Ne, Nt),
    reshape_moments(ui_arr,     _Ni, Nt),
    reshape_moments(Te_K,       _Ne, Nt),
    reshape_moments(Ti_K,       _Ni, Nt),
    reshape_moments(pe_arr,     _Ne, Nt),
    reshape_moments(pi_arr,     _Ni, Nt),
    reshape_moments(efract_arr, _Ne, Nt),
    reshape_moments(ifract_arr, _Ni, Nt),
    jnp.array(ion_z)[:, jnp.newaxis, jnp.newaxis],
    jnp.array(ion_a)[:, jnp.newaxis, jnp.newaxis],
    (wavelengths_nm * 1e-9)[jnp.newaxis, jnp.newaxis, :],
    probe_wavelength_nm * 1e-9,
    probe_vec, scatter_vec, ue_dir, ui_dir,
)

G_arr = np.asarray(_gain_mod.gain_factor(
    sd_out.sum_chiE, sd_out.sum_chiI, sd_out.epsilon,
    sd_out.k, sd_out.ks,
    scattering_angle=theta,
    probe_wavelength=probe_wavelength_nm * 1e-9,
    probe_intensity=probe_intensity,
    probe_diameter=probe_diameter,
    pol_p_fraction=pol_p_fraction,
    mode="exact",
))  # (Nk, Nt)


# ── Plot streaks side by side, plus gain factor ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

vmax_signal = float(max(Pkl_no_gain.max(), Pkl_gain.max()))

panels = [
    (Pkl_no_gain, "No gain"),
    (Pkl_gain,
     f"With gain  (I={probe_intensity:.0e} W/cm$^2$, "
     f"D={probe_diameter*1e6:.0f} $\\mu$m, p-pol, $\\theta_s$={scattering_angle_deg:.0f}$^\\circ$)"),
]
for ax, (Pkl, title) in zip(axes[:2], panels):
    im = ax.pcolormesh(
        t_ns, wavelengths_nm, Pkl,
        vmin=0.0, vmax=vmax_signal, shading="auto", cmap = "magma_r"
    )
    ax.axhline(probe_wavelength_nm, color="white", lw=0.5, ls="--", alpha=0.6)
    ax.set_xlabel("Time (ns)")
    ax.set_title(title)
axes[0].set_ylabel("Wavelength (nm)")
fig.colorbar(im, ax=axes[:2], label="Normalized scattered power",
             fraction=0.04, pad=0.02)

log_dev = float(np.nanmax(np.abs(np.log(G_arr))))
im2 = axes[2].pcolormesh(
    t_ns, wavelengths_nm, G_arr,
    norm=LogNorm(vmin=np.exp(-log_dev), vmax=np.exp(log_dev)),
    cmap="RdBu_r", shading="auto",
)
axes[2].axhline(probe_wavelength_nm, color="black", lw=0.5, ls="--", alpha=0.6)
axes[2].set_xlabel("Time (ns)")
axes[2].set_title("Gain factor $G$")
fig.colorbar(im2, ax=axes[2], label="$G$",
             fraction=0.04, pad=0.02)

fig.suptitle("Turnbull SRS/SBS gain correction — three-flow deuterium plasma",
             y=1.02)

out_path = "gain_check.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Wrote {out_path}")
plt.show()
