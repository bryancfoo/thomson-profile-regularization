"""
Test script for fitting Thomson scattered spectra against FLASH simulation results.

Reads synthetic simulation data from LaserSlab_FLASH.h5, which contains plasma
moments in hf["fields"][{field_name}] and coordinates in hf["coords"]["t"] and ["x"].
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import k as kB, e, epsilon_0

from forward import _scattered_power_wavelength

H5_FILE = "LaserSlab_FLASH.h5"

# ── Thomson geometry ────────────────────────────────────────────────────────
# 60-degree scattering angle; k vector aligned with x (velx direction).
# probe_vec · scatter_vec = cos(60°) = 0.5 ✓
# k ∝ scatter_vec - probe_vec = [1, 0, 0] ✓
PROBE_WAVELENGTH = 220e-9             # m
EPW_LAM = np.linspace(220e-9 - 20e-9, 220e-9 + 20e-9, 1024)  # m

PROBE_VEC   = np.array([-0.5, 0.0, np.sqrt(3) / 2])
SCATTER_VEC = np.array([ 0.5, 0.0, np.sqrt(3) / 2])
K_DIR       = np.array([ 1.0, 0.0, 0.0])          # unit vector along k

# ── Ion species ─────────────────────────────────────────────────────────────
Z_HE, A_HE = 2,  4.003   # helium-4
Z_AL, A_AL = 13, 26.982  # aluminum-27
M_U_G = 1.66054e-24      # atomic mass unit in grams

# ── Measurement x position ──────────────────────────────────────────────────
XX_UM = 0   # µm — red line in overview plots, lineout for Thomson streak


# ── I/O helpers ─────────────────────────────────────────────────────────────

def get_field(field_name: str) -> np.ndarray:
    """Return field from the FLASH HDF5 file, transposed to (x, t) order."""
    with h5py.File(H5_FILE, "r") as hf:
        return hf["fields"][field_name][:].T


def get_coords() -> tuple[np.ndarray, np.ndarray]:
    """Return (t [ns], x [µm]) coordinate arrays."""
    with h5py.File(H5_FILE, "r") as hf:
        t = hf["coords"]["t"][:] * 1e9     # s  → ns
        x = hf["coords"]["x"][:] * 1e4     # cm → µm
    return t, x


def get_lineout(field_name: str, xx_um: float) -> np.ndarray:
    """Return the time-resolved lineout of a field at a fixed x position.

    If xx_um falls between grid cells the value is linearly interpolated.

    Parameters
    ----------
    field_name : str
        Field to read.
    xx_um : float
        x position in µm.

    Returns
    -------
    np.ndarray, shape (Nt,)
    """
    _, x = get_coords()
    data = get_field(field_name)           # (Nx, Nt)
    return np.array([np.interp(xx_um, x, data[:, i])
                     for i in range(data.shape[1])])


# ── Thomson streak ───────────────────────────────────────────────────────────

def compute_thomson_streak(xx_um: float) -> np.ndarray:
    """Compute the synthetic Thomson EPW streak at x = xx_um.

    Returns
    -------
    np.ndarray, shape (Nk, Nt)
        Scattered power as a function of wavelength and time.
    """
    t, _ = get_coords()
    Nt = len(t)

    # --- raw FLASH lineouts ---
    dens = get_lineout("dens", xx_um)   # g/cm³
    targ = get_lineout("targ", xx_um)   # Al mass fraction
    cham = get_lineout("cham", xx_um)   # He mass fraction
    velx = get_lineout("velx", xx_um) * 1e-2   # cm/s → m/s
    tele = get_lineout("tele", xx_um)          # K
    tion = get_lineout("tion", xx_um)          # K

    # --- ion number densities (m^-3) — inputs to scattered_power stay SI ---
    n_Al = dens * targ / (A_AL * M_U_G) * 1e6
    n_He = dens * cham / (A_HE * M_U_G) * 1e6

    # --- electron number density (assuming full ionization) ---
    n_e = Z_AL * n_Al + Z_HE * n_He    # m^-3

    # --- charge fractions (must sum to 1 per timestep) ---
    ifract_He = Z_HE * n_He / n_e
    ifract_Al = Z_AL * n_Al / n_e

    Pkl = _scattered_power_wavelength(
        n               = n_e,                          # m^-3
        Te              = np.array([tele]),              # K
        Ti              = np.array([tion, tion]),        # K
        ue              = np.array([velx]),           # m/s
        ui              = np.array([velx, velx]),  # m/s
        pe              = np.ones((1, Nt)) * 2.0,
        pi              = np.ones((2, Nt)) * 2.0,
        efract          = np.ones((1, Nt)),
        ifract          = np.array([ifract_He, ifract_Al]),
        ion_z           = np.array([float(Z_HE), float(Z_AL)]),
        ion_a           = np.array([A_HE, A_AL]),
        wavelengths     = EPW_LAM,
        probe_wavelength= PROBE_WAVELENGTH,
        probe_vec       = PROBE_VEC,
        scatter_vec     = SCATTER_VEC,
        ue_dir          = K_DIR,
        ui_dir          = K_DIR,
        normalization_type="integral"
    )
    return np.array(Pkl)   # (Nk, Nt)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    t, x = get_coords()

    # ── Overview pcolormesh plots ───────────────────────────────────────────
    # scales convert raw FLASH field values to plot units
    K_TO_EV = kB / e
    fields = ["dens",    "velx",  "tele",      "tion"]
    labels = ["Density", "Vel. x", "T_e (eV)", "T_i (eV)"]
    cmaps  = ["magma_r", "bwr",   "hot",       "hot"]
    norms  = [LogNorm(), None,    None,         None]
    scales = [1.0,       1.0,     K_TO_EV,     K_TO_EV]

    _, axes = plt.subplots(ncols=4, figsize=(18, 5))
    for i, (ax, field_name, label, cmap, norm, scale) in enumerate(
            zip(axes, fields, labels, cmaps, norms, scales)):
        data = get_field(field_name) * scale
        pcm = ax.pcolormesh(t, x, data, shading="auto", cmap=cmap, norm=norm)
        plt.colorbar(pcm, ax=ax, label=label)
        ax.axhline(XX_UM, color="red", linewidth=1.0)
        ax.set_xlabel("t (ns)")
        ax.set_title(label)
        if i == 0:
            ax.set_ylabel("x (µm)")
        else:
            ax.set_yticklabels([])

    plt.tight_layout()

    # ── Time-resolved lineouts ─────────────────────────────────────────────
    dens_lo = get_lineout("dens", XX_UM)
    targ_lo = get_lineout("targ", XX_UM)
    cham_lo = get_lineout("cham", XX_UM)
    velx_lo = get_lineout("velx", XX_UM)
    tele_lo = get_lineout("tele", XX_UM)
    tion_lo = get_lineout("tion", XX_UM)

    n_Al = dens_lo * targ_lo / (A_AL * M_U_G) * 1e6   # m^-3
    n_He = dens_lo * cham_lo / (A_HE * M_U_G) * 1e6
    n_e  = Z_AL * n_Al + Z_HE * n_He

    # α = 1 / (k_scatt · λ_D); for 60° scattering k_scatt = 2π/λ_probe
    k_scatt  = 2 * np.pi / PROBE_WAVELENGTH
    lambda_D = np.sqrt(epsilon_0 * kB * tele_lo / (n_e * e**2))
    alpha    = 1.0 / (k_scatt * lambda_D)

    _, lo_axes = plt.subplots(nrows=2, ncols=4, figsize=(17, 7), sharex=True)
    lo_axes[0, 3].set_visible(False)   # only 7 panels needed

    lo_axes[0, 0].semilogy(t, n_e  * 1e-6);           lo_axes[0, 0].set_ylabel("n_e (cm⁻³)")
    lo_axes[0, 1].semilogy(t, n_He * 1e-6);           lo_axes[0, 1].set_ylabel("n_i He (cm⁻³)")
    lo_axes[0, 2].semilogy(t, n_Al * 1e-6);           lo_axes[0, 2].set_ylabel("n_i Al (cm⁻³)")
    lo_axes[1, 0].plot(t, tele_lo * K_TO_EV);         lo_axes[1, 0].set_ylabel("T_e (eV)")
    lo_axes[1, 1].plot(t, tion_lo * K_TO_EV);         lo_axes[1, 1].set_ylabel("T_i (eV)")
    lo_axes[1, 2].plot(t, velx_lo * 1e-2);            lo_axes[1, 2].set_ylabel("v_x (m/s)")
    lo_axes[1, 3].plot(t, alpha);                     lo_axes[1, 3].set_ylabel("α")
    lo_axes[1, 3].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    for ax in lo_axes[1]:
        ax.set_xlabel("t (ns)")
    for ax in lo_axes.flat:
        if ax.get_visible():
            ax.set_title(ax.get_ylabel())

    plt.suptitle(f"Lineouts at x = {XX_UM:.0f} µm")
    plt.tight_layout()

    # ── Thomson streak ──────────────────────────────────────────────────────
    Pkl = compute_thomson_streak(XX_UM)   # (Nk, Nt)

    _, ax2 = plt.subplots(figsize=(8, 5))
    lam_nm = EPW_LAM * 1e9
    notch = np.abs(lam_nm - PROBE_WAVELENGTH * 1e9) < 3.0
    Pkl_plot = np.where(notch[:, np.newaxis], np.nan, Pkl)
    pcm2 = ax2.pcolormesh(t, lam_nm, Pkl_plot, shading="auto", cmap="inferno")
    plt.colorbar(pcm2, ax=ax2, label="Scattered power (norm.)")
    ax2.set_xlabel("t (ns)")
    ax2.set_ylabel("Wavelength (nm)")
    ax2.set_title(f"Synthetic Thomson EPW streak  (x = {XX_UM:.0f} µm)")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
