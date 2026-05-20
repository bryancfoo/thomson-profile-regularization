"""Compute and plot time-resolved Thomson scattering from plasma profiles.

Usage
-----
    thomson-forward path/to/deck.toml        # after `pip install -e .`
    python thomson-forward.py path/to/deck.toml
    python thomson-forward.py                # falls back to interactive prompt

Takes an input deck (TOML) specifying plasma parameter profiles, computes the
forward-model Thomson scattering spectrum, and plots the time-resolved streak.
The deck schema includes sections: [profiles], [measurement], [plotting], and
[output].

See ``DECK_API.md`` at the repo root for the full deck schema.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k as kB, e

from ThomsonScattering.deck import load_deck, _load_array, _require
from ThomsonScattering.forward import scattered_power_wavelength


def build_settings_from_forward_deck(deck):
    """Convert a parsed forward deck into arguments for scattered_power_wavelength.

    Parameters
    ----------
    deck : dict
        Output of load_deck (includes private ``_base_dir`` and ``_deck_stem`` keys).

    Returns
    -------
    Pklam              : np.ndarray (Nk, Nt) Thomson scattered power
    time               : np.ndarray (Nt,) time axis
    wavelengths        : np.ndarray (Nk,) wavelength grid
    output_path        : pathlib.Path
    plotting_settings  : dict
    """
    base_dir = deck.get("_base_dir", Path("."))

    # ── 1. Load profile section ───────────────────────────────────────────────
    profiles_sec = deck.get("profiles", {})
    _require(
        profiles_sec,
        ["time", "ne", "Te", "Ti", "ue", "ui", "pe", "pi", "efract", "ifract"],
        section="[profiles]",
    )

    # Load time axis
    time = np.asarray(profiles_sec["time"])
    Nt = len(time)

    # Load density and temperatures
    n = np.asarray(profiles_sec["ne"])  # electron density in cm^-3
    Te = np.asarray(profiles_sec["Te"])  # electron temperature in eV
    Ti = np.asarray(profiles_sec["Ti"])  # ion temperatures, shape (Nions, Nt)

    # Load velocities
    ue = np.asarray(profiles_sec["ue"])
    ui = np.asarray(profiles_sec["ui"])  # shape (Nions, Nt)

    # Load distribution exponents
    pe = np.asarray(profiles_sec["pe"])
    pi = np.asarray(profiles_sec["pi"])  # shape (Nions, Nt)

    # Load species fractions
    efract = np.asarray(profiles_sec["efract"])
    ifract = np.asarray(profiles_sec["ifract"])  # shape (Nions, Nt)

    # ── 2. Build measurement_settings ─────────────────────────────────────────
    meas_raw = deck.get("measurement", {})
    _require(
        meas_raw,
        [
            "Nelectrons",
            "ion_z",
            "ion_a",
            "probe_wavelength",
            "probe_vec",
            "scatter_vec",
            "ue_dir",
            "ui_dir",
            "wavelengths",
        ],
        section="[measurement]",
    )

    # Fields that must become numpy arrays (inline list or file ref)
    _arr_fields = {
        "wavelengths",
        "instr_func_arr",
        "throughput",
        "aperture_weights",
        "scatter_vec",
        "probe_vec",
        "ue_dir",
        "ui_dir",
        "ion_z",
        "ion_a",
    }

    measurement_settings = {}
    for key, val in meas_raw.items():
        if key in _arr_fields:
            measurement_settings[key] = (
                _load_array(val, base_dir) if isinstance(val, str) else np.asarray(val)
            )
        elif key == "notch" and val is not None:
            measurement_settings[key] = tuple(val)
        else:
            measurement_settings[key] = val

    wavelengths = measurement_settings["wavelengths"]

    # Optional probe-beam parameters for SRS/SBS gain correction
    pb = deck.get("probe_beam")
    if pb is not None:
        mode = pb.get("gain_mode", "exact")
        if mode not in ("exact", "small_gain", "off"):
            raise ValueError(
                f"[probe_beam] gain_mode must be 'exact'|'small_gain'|'off', got {mode!r}"
            )
        fp = float(pb.get("pol_p_fraction", 1.0))
        if not (0.0 <= fp <= 1.0):
            raise ValueError(
                f"[probe_beam] pol_p_fraction must be in [0, 1], got {fp}"
            )
        measurement_settings["probe_intensity"] = float(pb["intensity_W_per_cm2"])
        measurement_settings["probe_diameter"] = float(pb["diameter_um"]) * 1e-6
        measurement_settings["pol_p_fraction"] = fp
        measurement_settings["gain_mode"] = mode

    # ── 3. Call forward model ────────────────────────────────────────────────
    Pklam = scattered_power_wavelength(
        n=n,
        ue=ue,
        ui=ui,
        Te=Te,
        Ti=Ti,
        pe=pe,
        pi=pi,
        efract=efract,
        ifract=ifract,
        ion_z=np.asarray(measurement_settings["ion_z"]),
        ion_a=np.asarray(measurement_settings["ion_a"]),
        wavelengths=wavelengths,
        probe_wavelength=measurement_settings["probe_wavelength"],
        probe_vec=np.asarray(measurement_settings["probe_vec"]),
        scatter_vec=np.asarray(measurement_settings["scatter_vec"]),
        ue_dir=np.asarray(measurement_settings["ue_dir"]),
        ui_dir=np.asarray(measurement_settings["ui_dir"]),
        instr_func_arr=measurement_settings.get("instr_func_arr"),
        irf_normalization=measurement_settings.get("irf_normalization", "area"),
        throughput=measurement_settings.get("throughput"),
        aperture_weights=measurement_settings.get("aperture_weights"),
        background_coefs=None,
        normalization_type=measurement_settings.get("normalization_type", "max"),
        normalization_scale=1,
        notch=measurement_settings.get("notch"),
        probe_intensity=measurement_settings.get("probe_intensity", 0.0),
        probe_diameter=measurement_settings.get("probe_diameter", 1.0),
        pol_p_fraction=measurement_settings.get("pol_p_fraction", 1.0),
        gain_mode=measurement_settings.get("gain_mode", "off"),
    )

    # ── 4. Get output and plotting settings ──────────────────────────────────
    output_raw = deck.get("output", {})
    out_rel = output_raw.get("path", None)
    if out_rel is None:
        stem = deck.get("_deck_stem", "forward")
        output_path = base_dir / f"{stem}_streak.png"
    else:
        output_path = base_dir / out_rel

    plotting_settings = deck.get("plotting", {})

    # Optional HDF5 output for results
    data_path = None
    if "data_path" in output_raw:
        data_path = base_dir / output_raw["data_path"]

    return Pklam, time, wavelengths, output_path, plotting_settings, data_path


def plot_streak(Pklam, time, wavelengths, output_path, plotting_settings=None):
    """Plot the time-resolved Thomson scattering streak.

    Parameters
    ----------
    Pklam : array (Nk, Nt)
        Scattered power spectrum
    time : array (Nt,)
        Time axis
    wavelengths : array (Nk,)
        Wavelength grid
    output_path : Path
        Output file path for the plot
    plotting_settings : dict
        Plotting options: figsize, dpi, cmap
    """
    if plotting_settings is None:
        plotting_settings = {}

    figsize = plotting_settings.get("figsize", [12, 6])
    dpi = plotting_settings.get("dpi", 150)
    cmap = plotting_settings.get("cmap", "viridis")

    fig, ax = plt.subplots(figsize=figsize)

    vmin = np.nanmin(Pklam)
    vmax = np.nanmax(Pklam)

    mesh = ax.pcolormesh(time, wavelengths, Pklam, vmin=vmin, vmax=vmax, shading="auto", cmap=cmap)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title("Time-Resolved Thomson Scattering Streak")

    cbar = fig.colorbar(mesh, ax=ax, label="Scattered Power (normalized)")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        deck_path = sys.argv[1]
    else:
        deck_path = input("Path to input deck (.toml): ").strip()

    if not deck_path:
        print("Error: no deck path provided")
        sys.exit(1)

    try:
        deck = load_deck(deck_path)
        Pklam, time, wavelengths, output_path, plotting_settings, data_path = (
            build_settings_from_forward_deck(deck)
        )

        # Plot the streak
        saved_path = plot_streak(Pklam, time, wavelengths, output_path, plotting_settings)
        print(f"Saved Thomson scattering streak to {saved_path}")

        # Optional: save data to HDF5
        if data_path is not None:
            import h5py

            data_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(data_path, "w") as fh:
                fh.create_dataset("Pklam", data=Pklam)
                fh.create_dataset("time", data=time)
                fh.create_dataset("wavelengths", data=wavelengths)
            print(f"Saved forward model data to {data_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
